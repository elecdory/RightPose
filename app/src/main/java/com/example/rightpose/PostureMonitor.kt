package com.example.rightpose

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions
import com.google.mlkit.vision.pose.PoseLandmark
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * PostureMonitor는 ML Kit Pose Detection API를 사용하여 카메라 프레임을 분석하고 자세를 평가합니다.
 *
 * @param context 애플리케이션 Context.
 */
class PostureMonitor(private val context: Context) {

    private val TAG = "PostureMonitor"

    enum class MonitorState {
        IDLE, CALIBRATING, MONITORING, ALERT_BAD_POSTURE, DROWSINESS_DETECTED, USER_AWAY
    }

    enum class PostureType {
        NORMAL, HEAD_FORWARD_TILT, HEAD_BACKWARD_TILT, HEAD_SIDE_TILT, HEAD_ROLL_TILT, SHOULDER_TILT, DROWSINESS, USER_AWAY
    }

    data class Angles(
        val headX: Float,
        val headY: Float,
        val headZ: Float,
        val shoulderAngle: Float,
        val eyeOpenness: Float
    )

    interface PostureListener {
        fun onStateChanged(state: MonitorState)
        fun onDebugInfo(debugInfo: String)
    }

    private var listener: PostureListener? = null
    private var currentState = MonitorState.IDLE
    private var bluetoothManager: BluetoothManager? = null

    // 사용자 부재 감지 관련 변수
    private var userAwayTimestamp: Long = 0
    private var lastUserPresenceTime: Long = 0
    private var userPresenceConfidence = 0f

    // 사용자 자리 이탈 감지 설정
    private val USER_AWAY_THRESHOLD_MS = 3000     // 3초 동안 사용자가 감지되지 않으면 자리 비움으로 처리 (기존 5초에서 단축)
    private val USER_AWAY_CONFIDENCE_THRESHOLD = -3f  // 자리 비움 확정을 위한 신뢰도 임계값 (기존 -5에서 조정)

    // 사용자 복귀 감지 설정
    private val USER_RETURN_CONFIRMATION_FRAMES = 2   // 사용자 복귀 확인을 위한 연속 프레임 수 (기존 3에서 조정)
    private val USER_RETURN_TIME_THRESHOLD_MS = 800  // 사용자 복귀 확인을 위한 시간 임계값 (기존 1000에서 조정)
    private val USER_RETURN_CONFIDENCE_THRESHOLD = 2f // 사용자 복귀 확정을 위한 신뢰도 임계값 (기존 3에서 조정)

    private var userPresentFramesCount = 0
    private var userAbsentFramesCount = 0
    private var lastDetectionTime = System.currentTimeMillis()

    // 추가: 연속 부재 프레임 카운트 임계값
    private val CONSECUTIVE_ABSENT_FRAMES_THRESHOLD = 10

    // 추가: 움직임 감지를 위한 변수
    private var lastPosePosition: Pair<Float, Float>? = null
    private var lastFacePosition: Pair<Float, Float>? = null
    private var positionHistory = mutableListOf<Pair<Float, Float>>()
    private val MOTION_HISTORY_SIZE = 5
    private val MOTION_THRESHOLD = 15f // 움직임으로 간주할 최소 픽셀 변화량

    // 추가: 환경 노이즈 보정을 위한 변수
    private var environmentalNoiseLevel = 0f
    private var noiseCalibrationCount = 0
    private val MAX_NOISE_CALIBRATION_COUNT = 30

    // 추가: 자세 랜드마크 가중치
    private val LANDMARK_WEIGHTS = mapOf(
        PoseLandmark.NOSE to 3.0f,
        PoseLandmark.LEFT_EYE to 2.5f,
        PoseLandmark.RIGHT_EYE to 2.5f,
        PoseLandmark.LEFT_EAR to 1.5f,
        PoseLandmark.RIGHT_EAR to 1.5f,
        PoseLandmark.LEFT_SHOULDER to 2.0f,
        PoseLandmark.RIGHT_SHOULDER to 2.0f,
        PoseLandmark.LEFT_ELBOW to 1.0f,
        PoseLandmark.RIGHT_ELBOW to 1.0f
    )

    // ML Kit PoseDetector 설정 (STREAM_MODE) - AccuratePoseDetectorOptions 사용
    private val options = AccuratePoseDetectorOptions.Builder()
        .setDetectorMode(AccuratePoseDetectorOptions.STREAM_MODE)
        .build()
    private val poseDetector: PoseDetector = PoseDetection.getClient(options)

    // 캘리브레이션 관련 변수
    private var isCalibrated = false
    private var calibrationAngles: Angles? = null

    fun initialize() {
        updateState(MonitorState.IDLE)
        logDebug("PostureMonitor 초기화 완료")
    }

    fun setListener(listener: PostureListener) {
        this.listener = listener
    }

    fun setBluetoothManager(manager: BluetoothManager) {
        this.bluetoothManager = manager
        logDebug("BluetoothManager 설정 완료")
    }

    fun startMonitoring() {
        Log.d(TAG, "모니터링 시작")
        updateState(MonitorState.MONITORING)
        lastUserPresenceTime = System.currentTimeMillis()
        userPresenceConfidence = 0f
        userPresentFramesCount = 0
        userAbsentFramesCount = 0
        // 추가: 환경 노이즈 캘리브레이션 초기화
        environmentalNoiseLevel = 0f
        noiseCalibrationCount = 0
        logDebug("자세 모니터링 시작됨")
    }

    fun processImage(face: Face?, pose: Pose?) {
        val currentTime = System.currentTimeMillis()
        val timeSinceLastDetection = currentTime - lastDetectionTime
        lastDetectionTime = currentTime

        // 추가: 움직임 감지
        val motionScore = detectMotion(face, pose)

        // 신뢰도 점수 계산 (향상된 버전)
        val presenceScore = calculateEnhancedPresenceScore(face, pose, motionScore, timeSinceLastDetection)

        // 환경 노이즈 보정
        if (noiseCalibrationCount < MAX_NOISE_CALIBRATION_COUNT) {
            // 초기 프레임에서 환경 노이즈 레벨 학습
            if (presenceScore < 0) {
                environmentalNoiseLevel = (environmentalNoiseLevel * noiseCalibrationCount + abs(presenceScore)) / (noiseCalibrationCount + 1)
                noiseCalibrationCount++
                logDebug("환경 노이즈 레벨 학습 중: $environmentalNoiseLevel ($noiseCalibrationCount/$MAX_NOISE_CALIBRATION_COUNT)")
            }
        }

        // 보정된 신뢰도 점수 계산
        val adjustedScore = if (presenceScore < 0 && abs(presenceScore) < environmentalNoiseLevel * 1.5f) {
            presenceScore * 0.5f // 환경 노이즈와 유사한 수준의 음수 점수는 영향력 감소
        } else {
            presenceScore
        }

        updateUserPresenceConfidence(adjustedScore, timeSinceLastDetection)

        // 사용자 존재 여부 결정 (향상된 알고리즘)
        val isUserPresent = isUserPresent(face, pose, adjustedScore)

        // 디버그 정보 기록
        logDebug("감지: 얼굴=${face != null}, 포즈=${pose != null}, 원점수=$presenceScore, 보정=$adjustedScore, 움직임=$motionScore, 신뢰도=$userPresenceConfidence")

        // 사용자 존재/부재 상태 처리 (향상된 알고리즘)
        processUserPresence(isUserPresent, currentTime)

        // 사용자가 존재할 경우 자세 분석
        if (isUserPresent && pose != null) {
            analyzePosture(face, pose)
        }
    }

    /**
     * 향상된 사용자 존재 신뢰도 점수 계산
     * 얼굴, 포즈, 움직임 정보를 종합적으로 분석
     */
    private fun calculateEnhancedPresenceScore(
        face: Face?,
        pose: Pose?,
        motionScore: Float,
        timeSinceLastDetection: Long
    ): Float {
        var score = 0f

        // 1. 얼굴 감지 점수 (가중치: 3)
        if (face != null) {
            score += 3f

            // 얼굴 회전 각도에 따른 점수 조정
            val headEulerX = abs(face.headEulerAngleX)
            val headEulerY = abs(face.headEulerAngleY)

            // 극단적 각도에 따른 감점 (개선된 로직)
            when {
                headEulerX > 45 || headEulerY > 45 -> score -= 2f
                headEulerX > 30 || headEulerY > 30 -> score -= 1f
                headEulerX > 20 || headEulerY > 20 -> score -= 0.5f
            }

            // 눈 감지 및 개방 여부 확인 (개선된 로직)
            val leftEyeOpen = face.leftEyeOpenProbability ?: 0f
            val rightEyeOpen = face.rightEyeOpenProbability ?: 0f
            val eyeOpenScore = (leftEyeOpen + rightEyeOpen) / 2

            when {
                eyeOpenScore > 0.7f -> score += 1.0f
                eyeOpenScore > 0.5f -> score += 0.7f
                eyeOpenScore > 0.3f -> score += 0.4f
                else -> score -= 0.2f // 눈을 감고 있을 가능성 (졸음일 수 있음)
            }

            // 얼굴 랜드마크 감지 품질 반영
            val trackingId = face.trackingId
            if (trackingId != null) {
                score += 0.5f // 얼굴 추적 ID가 있으면 더 신뢰할 수 있음
            }
        } else {
            score -= 2.5f
        }

        // 2. 포즈 감지 점수 (가중치: 2.5) - 개선된 로직
        if (pose != null) {
            val landmarks = pose.allPoseLandmarks

            // 감지된 랜드마크 수에 따른 기본 점수
            when {
                landmarks.size >= 15 -> score += 2.5f
                landmarks.size >= 10 -> score += 2.0f
                landmarks.size >= 5 -> score += 1.5f
                landmarks.size > 0 -> score += 0.5f
                else -> score -= 1.5f
            }

            // 주요 상체 랜드마크 확인 (가중치 적용)
            var landmarkScore = 0f
            var landmarkCount = 0

            LANDMARK_WEIGHTS.forEach { (landmarkType, weight) ->
                val landmark = pose.getPoseLandmark(landmarkType)
                if (landmark != null) {
                    landmarkScore += weight * landmark.inFrameLikelihood
                    landmarkCount++
                }
            }

            // 가중 평균 점수 계산
            if (landmarkCount > 0) {
                val weightedAvgScore = landmarkScore / LANDMARK_WEIGHTS.values.sum()
                score += weightedAvgScore * 2.0f
            }
        } else {
            score -= 3.0f
        }

        // 3. 움직임 점수 반영 (가중치: 1.5) - 새로운 특성
        score += motionScore * 1.5f

        // 4. 시간 지연에 따른 패널티 (오랜 시간 감지되지 않으면 점수 감소)
        if (timeSinceLastDetection > 500) {
            val timeDecay = (timeSinceLastDetection / 500).coerceAtMost(3).toFloat()
            score -= timeDecay * 0.5f
        }

        return score
    }

    /**
     * 움직임 감지 함수 - 연속된 프레임 간의 위치 변화를 추적
     * 반환값: 움직임 강도에 따른 점수 (0.0 ~ 1.0)
     */
    private fun detectMotion(face: Face?, pose: Pose?): Float {
        var motionScore = 0f
        val currentPosition = getCurrentPosition(face, pose)

        if (currentPosition != null) {
            // 새 위치를 기록
            positionHistory.add(currentPosition)
            if (positionHistory.size > MOTION_HISTORY_SIZE) {
                positionHistory.removeAt(0)
            }

            // 이전 위치와 현재 위치 비교
            val previousPosition = if (positionHistory.size > 1) positionHistory[positionHistory.size - 2] else null

            if (previousPosition != null) {
                val deltaX = abs(currentPosition.first - previousPosition.first)
                val deltaY = abs(currentPosition.second - previousPosition.second)
                val distance = sqrt(deltaX.pow(2) + deltaY.pow(2))

                // 거리에 따른 움직임 점수 계산
                motionScore = when {
                    distance > MOTION_THRESHOLD * 2 -> 1.0f  // 큰 움직임
                    distance > MOTION_THRESHOLD -> 0.7f      // 감지 가능한 움직임
                    distance > MOTION_THRESHOLD / 2 -> 0.3f  // 작은 움직임
                    else -> 0.1f                             // 미세한 움직임 또는 정지
                }
            }

            // 움직임 패턴 분석 (급격한 움직임과 미세한 움직임 구분)
            if (positionHistory.size >= 3) {
                var pattern = 0f
                for (i in 1 until positionHistory.size) {
                    val prev = positionHistory[i-1]
                    val curr = positionHistory[i]
                    val dx = curr.first - prev.first
                    val dy = curr.second - prev.second
                    pattern += sqrt(dx * dx + dy * dy)
                }

                // 일정한 움직임이 감지되면 점수 보너스
                pattern /= positionHistory.size - 1
                if (pattern > MOTION_THRESHOLD / 3 && pattern < MOTION_THRESHOLD) {
                    motionScore += 0.2f  // 일정한 작은 움직임은 사람이 존재할 가능성이 높음
                }
            }
        }

        return motionScore.coerceIn(0f, 1f)
    }

    /**
     * 현재 얼굴과 포즈의 대표 위치 계산
     */
    private fun getCurrentPosition(face: Face?, pose: Pose?): Pair<Float, Float>? {
        // 1. 얼굴 위치 확인
        if (face != null) {
            val bounds = face.boundingBox
            lastFacePosition = Pair(bounds.exactCenterX(), bounds.exactCenterY())
            return lastFacePosition
        }

        // 2. 포즈 위치 확인 (주요 랜드마크 사용)
        if (pose != null) {
            val nose = pose.getPoseLandmark(PoseLandmark.NOSE)
            if (nose != null) {
                lastPosePosition = Pair(nose.position.x, nose.position.y)
                return lastPosePosition
            }

            // 코가 없으면 다른 상체 랜드마크 평균 계산
            val visibleLandmarks = mutableListOf<Pair<Float, Float>>()

            // 주요 상체 랜드마크 확인
            val landmarks = listOf(
                PoseLandmark.LEFT_EYE, PoseLandmark.RIGHT_EYE,
                PoseLandmark.LEFT_EAR, PoseLandmark.RIGHT_EAR,
                PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER
            )

            for (lm in landmarks) {
                pose.getPoseLandmark(lm)?.let { landmark ->
                    visibleLandmarks.add(Pair(landmark.position.x, landmark.position.y))
                }
            }

            if (visibleLandmarks.isNotEmpty()) {
                val avgX = visibleLandmarks.sumOf { it.first.toDouble() } / visibleLandmarks.size
                val avgY = visibleLandmarks.sumOf { it.second.toDouble() } / visibleLandmarks.size
                lastPosePosition = Pair(avgX.toFloat(), avgY.toFloat())
                return lastPosePosition
            }
        }

        // 3. 최근 위치 반환 (얼굴/포즈가 모두 없는 경우)
        return lastFacePosition ?: lastPosePosition
    }

    /**
     * 사용자 존재 신뢰도 업데이트 (시간 가중치 적용)
     */
    private fun updateUserPresenceConfidence(presenceScore: Float, timeSinceLastDetection: Long) {
        // 시간 기반 가중치 계산 (오래된 프레임일수록 영향력 감소)
        val timeWeight = (1000.0 / (timeSinceLastDetection + 1000.0)).toFloat().coerceIn(0.1f, 0.9f)

        // 현재 신뢰도에 새 점수 누적 (시간 가중치 적용)
        userPresenceConfidence = (userPresenceConfidence * (0.7f * timeWeight)) +
                (presenceScore * (0.3f + (1.0f - timeWeight)))

        // 신뢰도 값 범위 제한 (-10 ~ 10)
        userPresenceConfidence = userPresenceConfidence.coerceIn(-10f, 10f)
    }

    /**
     * 향상된 사용자 존재 여부 판단
     */
    private fun isUserPresent(face: Face?, pose: Pose?, presenceScore: Float): Boolean {
        // 1. 현재 프레임 분석
        val hasStrongEvidence = face != null && face.trackingId != null && face.leftEyeOpenProbability != null
        val hasModerateEvidence = pose != null && pose.allPoseLandmarks.size >= 5

        // 2. 신뢰도 기반 판단
        val confidenceBasedPresence = userPresenceConfidence > 0

        // 3. 결과 판단 로직
        return when {
            // 강한 증거가 있으면 거의 확실하게 존재
            hasStrongEvidence -> true

            // 중간 증거와 양수 신뢰도가 있으면 존재 가능성 높음
            hasModerateEvidence && confidenceBasedPresence -> true

            // 신뢰도가 명확하게 높으면 증거가 약해도 존재 간주
            userPresenceConfidence > 3.0f -> true

            // 신뢰도가 명확하게 낮으면 증거가 있어도 부재 간주
            userPresenceConfidence < -5.0f -> false

            // 기본적으로는 신뢰도 기반 판단
            else -> confidenceBasedPresence
        }
    }

    /**
     * 향상된 사용자 존재/부재 상태 처리
     * 연속 프레임 카운팅, 시간 기반 및 신뢰도 기반 상태 전환
     */
    private fun processUserPresence(isUserPresent: Boolean, currentTime: Long) {
        if (isUserPresent) {
            // 사용자가 감지된 경우
            lastUserPresenceTime = currentTime
            userAbsentFramesCount = 0
            userPresentFramesCount++

            // USER_AWAY 상태이면서 사용자가 복귀한 경우 처리 (개선된 로직)
            if (currentState == MonitorState.USER_AWAY) {
                // 복귀 신뢰도 계산 (연속 프레임 수와 신뢰도 점수 모두 고려)
                val returnConfidence = userPresentFramesCount * (userPresenceConfidence / 5.0f)

                if (userPresentFramesCount >= USER_RETURN_CONFIRMATION_FRAMES &&
                    (returnConfidence >= 1.0f || currentTime - lastUserPresenceTime < USER_RETURN_TIME_THRESHOLD_MS)) {

                    logDebug("사용자 복귀 확인됨 (${userPresentFramesCount}프레임, 신뢰도: $userPresenceConfidence, 복귀신뢰도: $returnConfidence)")
                    updateState(MonitorState.MONITORING)
                    userAwayTimestamp = 0
                    userPresentFramesCount = 0
                }
            }
        } else {
            // 사용자가 감지되지 않은 경우
            userPresentFramesCount = 0
            userAbsentFramesCount++

            // 첫 번째 부재 감지 시 타임스탬프 기록
            if (userAwayTimestamp == 0L) {
                userAwayTimestamp = currentTime
                logDebug("사용자 부재 감지 시작 (타임스탬프: $userAwayTimestamp)")
            }

            // 개선된 자리비움 판단 로직
            // 1. 시간 기반: 충분한 시간 동안 사용자가 감지되지 않음
            val timeAwaySufficient = currentTime - userAwayTimestamp > USER_AWAY_THRESHOLD_MS

            // 2. 프레임 기반: 연속된 부재 프레임이 임계값 초과
            val framesAwaySufficient = userAbsentFramesCount >= CONSECUTIVE_ABSENT_FRAMES_THRESHOLD

            // 3. 신뢰도 기반: 신뢰도가 임계값 이하
            val confidenceBelowThreshold = userPresenceConfidence <= USER_AWAY_CONFIDENCE_THRESHOLD

            // 조합 로직: 시간 + (프레임 또는 신뢰도)
            if (timeAwaySufficient && (framesAwaySufficient || confidenceBelowThreshold) &&
                currentState != MonitorState.USER_AWAY) {

                logDebug("사용자 자리 비움 확정: ${(currentTime - userAwayTimestamp) / 1000}초, " +
                        "연속 부재 프레임: $userAbsentFramesCount, 신뢰도: $userPresenceConfidence")
                updateState(MonitorState.USER_AWAY)
            }
        }
    }

    /**
     * 포즈와 얼굴 데이터 기반으로 자세 분석
     */
    private fun analyzePosture(face: Face?, pose: Pose?) {
        if (face == null || pose == null) return

        try {
            // 머리 각도 계산
            val headX = face.headEulerAngleX
            val headY = face.headEulerAngleY
            val headZ = face.headEulerAngleZ

            // 어깨 각도 계산
            var shoulderAngle = 0f
            val leftShoulder = pose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER)
            val rightShoulder = pose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER)

            if (leftShoulder != null && rightShoulder != null) {
                val deltaY = rightShoulder.position.y - leftShoulder.position.y
                val deltaX = rightShoulder.position.x - leftShoulder.position.x
                val angleRadians = kotlin.math.atan2(deltaY, deltaX)
                shoulderAngle = (angleRadians * (180 / kotlin.math.PI)).toFloat()
            }

            // 눈 개방도 계산
            val leftEyeOpen = face.leftEyeOpenProbability ?: 0.5f
            val rightEyeOpen = face.rightEyeOpenProbability ?: 0.5f
            val eyeOpenness = (leftEyeOpen + rightEyeOpen) / 2f

            // 각도 데이터 생성
            val angles = Angles(
                headX = headX,
                headY = headY,
                headZ = headZ,
                shoulderAngle = shoulderAngle,
                eyeOpenness = eyeOpenness
            )

            // 캘리브레이션이 필요한 경우 수행
            if (!isCalibrated) {
                calibratePosture(angles)
            }

            // 자세 평가
            evaluatePosture(angles)

        } catch (e: Exception) {
            Log.e(TAG, "자세 분석 중 오류 발생", e)
        }
    }

    /**
     * 자세 캘리브레이션 수행
     */
    private fun calibratePosture(angles: Angles) {
        calibrationAngles = angles
        isCalibrated = true
        logDebug("자세 캘리브레이션 완료: $angles")
    }

    /**
     * 자세 평가
     */
    private fun evaluatePosture(angles: Angles) {
        // 캘리브레이션 값이 없으면 기본값 사용
        val baseAngles = calibrationAngles ?: Angles(0f, 0f, 0f, 0f, 1f)

        // 자세 평가 로직 (실제 구현에서는 더 복잡한 로직 필요)
        val headXDeviation = abs(angles.headX - baseAngles.headX)
        val headYDeviation = abs(angles.headY - baseAngles.headY)
        val headZDeviation = abs(angles.headZ - baseAngles.headZ)
        val shoulderDeviation = abs(angles.shoulderAngle - baseAngles.shoulderAngle)

        // 자세 문제 체크 (임계값은 조정 필요)
        val hasHeadXIssue = headXDeviation > 15f
        val hasHeadYIssue = headYDeviation > 10f
        val hasHeadZIssue = headZDeviation > 10f
        val hasShoulderIssue = shoulderDeviation > 8f
        val hasDrowsinessIssue = angles.eyeOpenness < 0.3f

        // 디버그 로깅
        if (hasHeadXIssue || hasHeadYIssue || hasHeadZIssue || hasShoulderIssue || hasDrowsinessIssue) {
            logDebug("자세 문제 감지: " +
                    "머리X=${if (hasHeadXIssue) "불량" else "정상"}, " +
                    "머리Y=${if (hasHeadYIssue) "불량" else "정상"}, " +
                    "머리Z=${if (hasHeadZIssue) "불량" else "정상"}, " +
                    "어깨=${if (hasShoulderIssue) "불량" else "정상"}, " +
                    "졸음=${if (hasDrowsinessIssue) "감지" else "정상"}")
        }
    }

    /**
     * 상태 업데이트
     */
    private fun updateState(newState: MonitorState) {
        if (currentState != newState) {
            currentState = newState
            listener?.onStateChanged(newState)
            logDebug("상태 변경: $newState")
        }
    }

    /**
     * 디버그 정보 로깅
     */
    private fun logDebug(message: String) {
        Log.d(TAG, message)
        listener?.onDebugInfo(message)
    }

    fun analyzeFrame(imageProxy: ImageProxy) {
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)
            poseDetector.process(inputImage)
                .addOnSuccessListener { pose: Pose ->
                    val isUserPresent = isPoseValid(pose)
                    checkUserPresence(isUserPresent)

                    if (!isUserPresent) {
                        logDebug("유효한 포즈 감지되지 않음 - 사용자 없음")
                        imageProxy.close()
                        return@addOnSuccessListener
                    }

                    // 사용자가 존재하고 유효한 포즈가 감지된 경우
                    val angles = calculateAngles(pose, null)
                    val isCorrect = isPostureCorrect(angles)
                    if (isCorrect) {
                        logDebug("자세가 올바릅니다.")
                    } else {
                        logDebug("자세 교정이 필요합니다.")
                    }
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "분석 실패: ${e.message}")
                    logDebug("분석 실패: ${e.message}")
                    checkUserPresence(false)  // 분석 실패 시 사용자 부재 가능성 체크
                }
                .addOnCompleteListener {
                    imageProxy.close()
                }
        } else {
            imageProxy.close()
            checkUserPresence(false)  // 이미지가 없는 경우 사용자 부재 가능성 체크
        }
    }

    // 유효한 포즈가 감지되었는지 확인 (사용자 존재 여부)
    private fun isPoseValid(pose: Pose): Boolean {
        // 필수 랜드마크(눈, 코, 귀, 어깨 등)가 일정 개수 이상 감지되는지 확인
        val landmarks = pose.allPoseLandmarks
        if (landmarks.size < 5) {
            logDebug("감지된 랜드마크 수 부족: ${landmarks.size}")
            return false  // 최소 랜드마크 수 미달 시 사용자 없음
        }

        // 주요 상체 랜드마크가 감지되었는지 확인
        val nose = pose.getPoseLandmark(PoseLandmark.NOSE)
        val leftEye = pose.getPoseLandmark(PoseLandmark.LEFT_EYE)
        val rightEye = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE)
        val leftShoulder = pose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER)
        val rightShoulder = pose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER)

        val hasRequiredLandmarks = nose != null && leftEye != null && rightEye != null &&
                leftShoulder != null && rightShoulder != null

        if (!hasRequiredLandmarks) {
            logDebug("필수 랜드마크 누락: " +
                    "코=${nose != null}, 왼쪽눈=${leftEye != null}, 오른쪽눈=${rightEye != null}, " +
                    "왼쪽어깨=${leftShoulder != null}, 오른쪽어깨=${rightShoulder != null}")
        }

        return hasRequiredLandmarks
    }

    // 사용자 존재 여부 확인 및 상태 업데이트
    private fun checkUserPresence(isUserPresent: Boolean) {
        val currentTime = System.currentTimeMillis()

        if (!isUserPresent) {
            // 사용자가 감지되지 않는 경우
            if (userAwayTimestamp == 0L) {
                userAwayTimestamp = currentTime
                logDebug("사용자 부재 감지 시작 (타임스탬프: $userAwayTimestamp)")
            } else {
                val awayDuration = currentTime - userAwayTimestamp

                // 부재 지속 시간에 따른 신뢰도 감소
                val confidenceDecay = (awayDuration / 1000f).coerceAtMost(5f)
                userPresenceConfidence -= confidenceDecay * 0.5f
                userPresenceConfidence = userPresenceConfidence.coerceAtLeast(-10f)

                if (awayDuration > USER_AWAY_THRESHOLD_MS &&
                    currentState != MonitorState.USER_AWAY) {
                    updateState(MonitorState.USER_AWAY)
                    logDebug("사용자 자리 비움 문제로 확정 (${awayDuration / 1000}초, 신뢰도: $userPresenceConfidence)")
                }
            }
            userPresentFramesCount = 0  // 사용자 존재 프레임 카운트 리셋
            userAbsentFramesCount++
        } else {
            // 사용자가 감지된 경우
            userAbsentFramesCount = 0

            // 신뢰도 점수 증가 (복귀 확인용)
            userPresenceConfidence += 0.5f
            userPresenceConfidence = userPresenceConfidence.coerceAtMost(10f)

            if (currentState == MonitorState.USER_AWAY) {
                userPresentFramesCount++
                logDebug("사용자 복귀 감지 중 ($userPresentFramesCount/$USER_RETURN_CONFIRMATION_FRAMES, 신뢰도: $userPresenceConfidence)")

                if (userPresentFramesCount >= USER_RETURN_CONFIRMATION_FRAMES) {
                    updateState(MonitorState.MONITORING)
                    logDebug("사용자 자리비움 상태에서 회복됨")
                    userPresentFramesCount = 0
                }
            }
            userAwayTimestamp = 0  // 타임스탬프 리셋
            lastUserPresenceTime = currentTime
        }
    }

    // 포즈 데이터로부터 각도 계산
    private fun calculateAngles(pose: Pose, face: Face?): Angles {
        try {
            // 머리 기울기 계산 (x, y, z 축)
            val nose = pose.getPoseLandmark(PoseLandmark.NOSE)
            val leftEye = pose.getPoseLandmark(PoseLandmark.LEFT_EYE)
            val rightEye = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE)
            val leftEar = pose.getPoseLandmark(PoseLandmark.LEFT_EAR)
            val rightEar = pose.getPoseLandmark(PoseLandmark.RIGHT_EAR)

            // 어깨 기울기 계산
            val leftShoulder = pose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER)
            val rightShoulder = pose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER)

            // 눈 개방도 (눈의 y 좌표 차이로 대략적인 값 추정)
            var eyeOpenness = 0.8f
            if (face != null && face.leftEyeOpenProbability != null && face.rightEyeOpenProbability != null) {
                eyeOpenness = (face.leftEyeOpenProbability!! + face.rightEyeOpenProbability!!) / 2
            }

            var headX = 0f
            var headY = 0f
            var headZ = 0f
            var shoulderAngle = 0f

            // 머리 앞뒤 기울기(X축) 계산
            if (nose != null && (leftEar != null || rightEar != null)) {
                // 코와 귀의 좌표로 앞뒤 기울기 추정 (x,y 좌표 기반)
                val earPos = if (leftEar != null) leftEar.position else rightEar!!.position

                // x,y 좌표 기반 거리 계산으로 대체
                val distanceDiff = sqrt(
                    (nose.position.x - earPos.x).pow(2) +
                            (nose.position.y - earPos.y).pow(2)
                )
                headX = distanceDiff * 0.5f // 스케일 조정
            }

            // 머리 좌우 기울기(Y축) 계산
            if (leftEye != null && rightEye != null) {
                // 두 눈의 y좌표 차이로 좌우 기울기 추정
                val eyeYDiff = leftEye.position.y - rightEye.position.y
                headY = eyeYDiff * 5 // 스케일 조정
            }

            // 머리 회전(Z축) 계산
            if (leftEye != null && rightEye != null) {
                val eyeXDiff = leftEye.position.x - rightEye.position.x
                val eyeYDiff = leftEye.position.y - rightEye.position.y
                headZ = kotlin.math.atan2(eyeYDiff, eyeXDiff) * (180 / kotlin.math.PI).toFloat()
            }

            // 어깨 기울기 계산
            if (leftShoulder != null && rightShoulder != null) {
                val deltaY = rightShoulder.position.y - leftShoulder.position.y
                val deltaX = rightShoulder.position.x - leftShoulder.position.x
                val angleRadians = kotlin.math.atan2(deltaY, deltaX)
                shoulderAngle = (angleRadians * (180 / kotlin.math.PI)).toFloat()
            }

            return Angles(
                headX = headX,
                headY = headY,
                headZ = headZ,
                shoulderAngle = shoulderAngle,
                eyeOpenness = eyeOpenness
            )
        } catch (e: Exception) {
            Log.e(TAG, "각도 계산 중 오류: ${e.message}")
            return Angles(0f, 0f, 0f, 0f, 1f)  // 오류 시 기본값 반환
        }
    }

    // 자세가 올바른지 평가
    private fun isPostureCorrect(angles: Angles): Boolean {
        // 자세 평가 기준 (임계값은 실제 사용 사례에 맞게 조정해야 함)
        val isHeadXOk = abs(angles.headX) < 20f  // 머리 앞뒤 기울기
        val isHeadYOk = abs(angles.headY) < 15f  // 머리 좌우 기울기
        val isHeadZOk = abs(angles.headZ) < 10f  // 머리 회전
        val isShoulderOk = abs(angles.shoulderAngle) < 10f  // 어깨 기울기
        val isAwake = angles.eyeOpenness > 0.5f  // 눈 개방도 (졸음 감지)

        // 모든 조건을 충족해야 올바른 자세로 판단
        return isHeadXOk && isHeadYOk && isHeadZOk && isShoulderOk && isAwake
    }

    /**
     * 리소스 해제
     */
    fun release() {
        poseDetector.close()
        logDebug("PostureMonitor 리소스 해제됨")
    }
}