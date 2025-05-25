package com.example.rightpose

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Size
import android.util.TypedValue
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Observer
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), PostureMonitor.PostureListener, BluetoothManager.BluetoothManagerListener {

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN
        )
        private val REQUIRED_PERMISSIONS_S_PLUS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN,
            Manifest.permission.BLUETOOTH_CONNECT,
            Manifest.permission.BLUETOOTH_SCAN
        )
    }

    private lateinit var cameraExecutor: java.util.concurrent.ExecutorService
    private lateinit var viewFinder: PreviewView
    private lateinit var postureMonitor: PostureMonitor
    private lateinit var bluetoothManager: BluetoothManager

    // UI 요소
    private lateinit var statusTextView: TextView
    private lateinit var btStatusTextView: TextView
    private lateinit var postureIndicatorView: ImageView
    private lateinit var debugTextView: TextView
    private lateinit var showAnalysisInfoButton: Button

    // 슬라이딩 바텀 시트
    private lateinit var bottomSheetLayout: LinearLayout
    private lateinit var bottomSheetBehavior: BottomSheetBehavior<LinearLayout>
    private lateinit var menuHandleView: View

    // ML Kit 객체
    private lateinit var faceDetector: com.google.mlkit.vision.face.FaceDetector
    private lateinit var poseDetector: PoseDetector

    // 카메라 변수
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalyzer: ImageAnalysis? = null

    // 감지된 얼굴과 자세
    private var lastFace: Face? = null
    private var lastPose: Pose? = null

    // 자세 분석 정보
    private var currentPostureState = "정상"
    private var currentPostureType = PostureMonitor.PostureType.NORMAL
    private var postureHistoryList = mutableListOf<PostureHistoryItem>()
    private var lastPostureChangeTime = System.currentTimeMillis()
    private var headAngleX = 0f
    private var headAngleY = 0f
    private var headAngleZ = 0f
    private var shoulderAngle = 0f
    private var eyeOpenness = 0f

    // 자세 문제 지속 시간 추적을 위한 변수
    private val postureIssueStartTimes = mutableMapOf<PostureMonitor.PostureType, Long>()
    private val postureDurationThresholds = mapOf(
        PostureMonitor.PostureType.HEAD_FORWARD_TILT to 1500L,    // 머리 앞으로 기울임: 1.5초
        PostureMonitor.PostureType.HEAD_BACKWARD_TILT to 1500L,   // 머리 뒤로 기울임: 1.5초
        PostureMonitor.PostureType.HEAD_SIDE_TILT to 1500L,       // 머리 옆으로 기울임: 1.5초
        PostureMonitor.PostureType.HEAD_ROLL_TILT to 1500L,       // 머리 회전: 1.5초
        PostureMonitor.PostureType.SHOULDER_TILT to 2000L,        // 어깨 기울임: 2초
        PostureMonitor.PostureType.DROWSINESS to 2500L,           // 졸음: 2.5초
        PostureMonitor.PostureType.USER_AWAY to 3000L             // 자리비움: 3초
    )

    // 자세 문제 목록을 저장할 데이터 클래스
    data class PostureIssue(
        val type: PostureMonitor.PostureType,
        val description: String
    )

    // 현재 감지된 자세 문제들을 저장하는 리스트
    private val currentPostureIssues = mutableListOf<PostureIssue>()

    // 자세 기록 클래스
    data class PostureHistoryItem(
        val timestamp: Long,
        val postureType: PostureMonitor.PostureType,
        val description: String
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 뷰 바인딩
        viewFinder = findViewById(R.id.viewFinder)
        statusTextView = findViewById(R.id.statusTextView)
        btStatusTextView = findViewById(R.id.bluetoothStatusTextView)
        postureIndicatorView = findViewById(R.id.postureIndicatorView)
        debugTextView = findViewById(R.id.debugTextView)
        showAnalysisInfoButton = findViewById(R.id.showAnalysisInfoButton)

        // 바텀 시트 설정
        bottomSheetLayout = findViewById(R.id.bottomSheetLayout)
        menuHandleView = findViewById(R.id.menuHandleView)

        setupBottomSheet()
        enhanceTopFonts()

        // 디버그 텍스트뷰 표시 설정
        debugTextView.visibility = View.VISIBLE

        // 버튼 리스너 설정
        findViewById<Button>(R.id.calibrateButton).setOnClickListener {
            Log.d(TAG, "모니터링 시작 버튼 클릭")
            postureMonitor.startMonitoring()
            showToast("자세 모니터링이 시작되었습니다.")
        }

        findViewById<Button>(R.id.switchCameraButton).setOnClickListener {
            switchCamera()
        }

        findViewById<Button>(R.id.connectBluetoothButton).setOnClickListener {
            connectBluetooth()
        }

        // 자세 분석 정보 보기 버튼 리스너
        showAnalysisInfoButton.setOnClickListener {
            showPostureAnalysisDialog()
        }

        setupMLKit()
        setupPostureMonitor()
        setupBluetoothManager()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, getRequiredPermissions(), REQUEST_CODE_PERMISSIONS
            )
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        // 앱 시작 후 2초 후 자동으로 모니터링 시작
        Handler(Looper.getMainLooper()).postDelayed({
            Log.d(TAG, "앱 시작 후 자동으로 모니터링 시작")
            postureMonitor.startMonitoring()
            updateStatusUI(PostureMonitor.MonitorState.MONITORING)
        }, 2000)
    }

    // 자세 문제 추가 메서드
    private fun addPostureIssue(type: PostureMonitor.PostureType, description: String) {
        val currentTime = System.currentTimeMillis()

        // 자세 문제가 처음 감지되는 경우 시작 시간 기록
        if (!postureIssueStartTimes.containsKey(type)) {
            postureIssueStartTimes[type] = currentTime
            Log.d(TAG, "자세 문제 감지 시작: $description")
            return
        }

        // 이미 감지 중인 경우, 지속 시간 확인
        val startTime = postureIssueStartTimes[type] ?: currentTime
        val duration = currentTime - startTime
        val threshold = postureDurationThresholds[type] ?: 2000L

        // 임계값을 넘어서면 실제 자세 문제로 등록
        if (duration >= threshold) {
            if (!currentPostureIssues.any { it.type == type }) {
                val existingIndex = currentPostureIssues.indexOfFirst { it.type == type }
                if (existingIndex >= 0) {
                    currentPostureIssues[existingIndex] = PostureIssue(type, description)
                } else {
                    currentPostureIssues.add(PostureIssue(type, description))
                    Log.d(TAG, "자세 문제 추가: $description (${duration}ms 지속)")
                }
                updatePostureStatusFromIssues()
            }
        }
    }

    // 자세 문제 제거 메서드
    private fun removePostureIssue(type: PostureMonitor.PostureType) {
        postureIssueStartTimes.remove(type)

        // 자세 문제가 표시되고 있었던 경우에만 삭제 처리
        if (currentPostureIssues.any { it.type == type }) {
            currentPostureIssues.removeIf { it.type == type }
            updatePostureStatusFromIssues()
            Log.d(TAG, "$type 자세 문제 제거됨")
        }
    }

    // 감지된 모든 자세 문제에 기반하여 자세 상태를 업데이트
    private fun updatePostureStatusFromIssues() {
        if (currentPostureIssues.isEmpty()) {
            currentPostureType = PostureMonitor.PostureType.NORMAL
            currentPostureState = "자세가 바릅니다"
            postureIndicatorView.setColorFilter(android.graphics.Color.GREEN)
            bluetoothManager.sendCommand(BluetoothManager.CMD_NORMAL)
            Log.d(TAG, "블루투스 명령 전송: 정상 모드 (LED 꺼짐)")
        } else {
            val primaryIssue = currentPostureIssues.first()
            currentPostureType = primaryIssue.type
            val issueDescriptions = currentPostureIssues.joinToString(", ") { it.description }
            currentPostureState = "잘못된 자세: $issueDescriptions"
            postureIndicatorView.setColorFilter(android.graphics.Color.RED)
            sendBluetoothCommandForPostureIssues()
        }
        val currentTime = System.currentTimeMillis()
        postureHistoryList.add(PostureHistoryItem(currentTime, currentPostureType, currentPostureState))
        if (postureHistoryList.size > 20) {
            postureHistoryList.removeAt(0)
        }
        runOnUiThread {
            statusTextView.text = currentPostureState
            Log.d(TAG, "UI 상태 업데이트: $currentPostureState")
        }
    }

    // 블루투스 명령 전송 메서드
    private fun sendBluetoothCommandForPostureIssues() {
        if (!bluetoothManager.isConnected()) {
            Log.w(TAG, "블루투스가 연결되지 않았습니다. 명령을 전송할 수 없습니다.")
            showToast("독서대가 연결되지 않아 알림을 전송할 수 없습니다.")
            Handler(Looper.getMainLooper()).postDelayed({
                bluetoothManager.initializeBluetooth()
            }, 1000)
            return
        }
        val commandHistory = bluetoothManager.getCommandHistory()
        val lastCommand = if (commandHistory.isNotEmpty()) commandHistory.last() else ""
        if (currentPostureIssues.any { it.type == PostureMonitor.PostureType.DROWSINESS }) {
            val level = (currentPostureIssues.size + 1).coerceAtMost(5)
            val command = if (level > 1) {
                "${BluetoothManager.CMD_DROWSINESS_ALERT}_L$level"
            } else {
                BluetoothManager.CMD_DROWSINESS_ALERT
            }
            if (command != lastCommand) {
                bluetoothManager.sendCommand(command)
                Log.d(TAG, "블루투스 명령 전송: 졸음 경고 레벨 $level")
            }
            return
        }
        if (currentPostureIssues.any { it.type == PostureMonitor.PostureType.USER_AWAY }) {
            val level = (currentPostureIssues.size + 1).coerceAtMost(5)
            val command = if (level > 1) {
                "${BluetoothManager.CMD_FOCUS_ALERT}_L$level"
            } else {
                BluetoothManager.CMD_FOCUS_ALERT
            }
            if (command != lastCommand) {
                bluetoothManager.sendCommand(command)
                Log.d(TAG, "블루투스 명령 전송: 자리비움 경고 레벨 $level")
            }
            return
        }
        val postureLevel = when {
            currentPostureIssues.size >= 3 -> 5
            currentPostureIssues.size == 2 -> 3
            else -> 1
        }
        val command = when (postureLevel) {
            1 -> BluetoothManager.CMD_POSTURE_ALERT_L1
            3 -> BluetoothManager.CMD_POSTURE_ALERT_L3
            5 -> BluetoothManager.CMD_POSTURE_ALERT_L5
            else -> BluetoothManager.CMD_POSTURE_ALERT
        }
        if (command != lastCommand) {
            bluetoothManager.sendCommand(command)
            Log.d(TAG, "블루투스 명령 전송: 자세 경고 레벨 $postureLevel")
        }
    }

    // 자세 분석 정보 대화상자 표시
    private fun showPostureAnalysisDialog() {
        val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        val currentInfo = StringBuilder()
        currentInfo.append("현재 자세: $currentPostureState\n\n")
        currentInfo.append("머리 각도 (X, Y, Z): ${headAngleX.toInt()}°, ${headAngleY.toInt()}°, ${headAngleZ.toInt()}°\n")
        currentInfo.append("어깨 각도: ${shoulderAngle.toInt()}°\n")
        currentInfo.append("눈 열림 정도: ${(eyeOpenness * 100).toInt()}%\n\n")
        currentInfo.append("==== 자세 변화 기록 ====\n")
        if (postureHistoryList.isEmpty()) {
            currentInfo.append("자세 변화 기록이 없습니다.")
        } else {
            val recentHistory = postureHistoryList.takeLast(5).reversed()
            for (item in recentHistory) {
                val time = dateFormat.format(Date(item.timestamp))
                currentInfo.append("$time: ${item.description}\n")
            }
        }
        AlertDialog.Builder(this)
            .setTitle("자세 분석 정보")
            .setMessage(currentInfo.toString())
            .setPositiveButton("확인", null)
            .show()
    }

    // 바텀 시트 설정 함수
    private fun setupBottomSheet() {
        bottomSheetBehavior = BottomSheetBehavior.from(bottomSheetLayout)
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
        menuHandleView.setOnClickListener {
            bottomSheetBehavior.state = if (bottomSheetBehavior.state == BottomSheetBehavior.STATE_COLLAPSED)
                BottomSheetBehavior.STATE_EXPANDED
            else
                BottomSheetBehavior.STATE_COLLAPSED
        }
        bottomSheetBehavior.addBottomSheetCallback(object : BottomSheetBehavior.BottomSheetCallback() {
            override fun onStateChanged(bottomSheet: View, newState: Int) { }
            override fun onSlide(bottomSheet: View, slideOffset: Float) {
                menuHandleView.rotation = slideOffset * 180
            }
        })
    }

    // 상단 글꼴 강화 함수
    private fun enhanceTopFonts() {
        statusTextView.apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(ContextCompat.getColor(context, android.R.color.white))
            setShadowLayer(3f, 1f, 1f, android.graphics.Color.BLACK)
            setBackgroundResource(R.drawable.text_bg_rounded)
            setPadding(16, 8, 16, 8)
        }
        btStatusTextView.apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
            setTypeface(typeface, Typeface.BOLD)
            setShadowLayer(2f, 1f, 1f, android.graphics.Color.BLACK)
            setPadding(16, 8, 16, 8)
        }
    }

    private fun getRequiredPermissions(): Array<String> {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
            REQUIRED_PERMISSIONS_S_PLUS
        } else {
            REQUIRED_PERMISSIONS
        }
    }

    private fun setupMLKit() {
        val faceDetectorOptions = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .enableTracking()
            .build()
        faceDetector = FaceDetection.getClient(faceDetectorOptions)
        val poseDetectorOptions = AccuratePoseDetectorOptions.Builder()
            .setDetectorMode(AccuratePoseDetectorOptions.STREAM_MODE)
            .build()
        poseDetector = PoseDetection.getClient(poseDetectorOptions)
    }

    private fun setupPostureMonitor() {
        Log.d(TAG, "PostureMonitor 설정 시작")
        postureMonitor = PostureMonitor(this)
        postureMonitor.setListener(this)
        postureMonitor.initialize()
        Log.d(TAG, "PostureMonitor 초기화 완료")
    }

    private fun setupBluetoothManager() {
        bluetoothManager = BluetoothManager
        bluetoothManager.setBluetoothListener(this)
        if (!bluetoothManager.checkPermissions(this)) {
            showToast("블루투스 권한이 필요합니다")
        }
        postureMonitor.setBluetoothManager(bluetoothManager)
        bluetoothManager.connectionState.observe(this, Observer { state ->
            updateBluetoothStatusUI(state)
        })
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed")
        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()
        val preview = Preview.Builder()
            .setTargetResolution(Size(640, 480))
            .build()
        preview.setSurfaceProvider(viewFinder.surfaceProvider)
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build().also {
                it.setAnalyzer(cameraExecutor) { imageProxy: ImageProxy ->
                    processImage(imageProxy)
                }
            }
        try {
            cameraProvider.unbindAll()
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )
            Log.d(TAG, "카메라 바인딩 성공")
        } catch (exc: Exception) {
            Log.e(TAG, "카메라 바인딩 실패", exc)
        }
    }

    private fun processImage(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image ?: run {
            imageProxy.close()
            return
        }
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val image = InputImage.fromMediaImage(mediaImage, rotationDegrees)
        faceDetector.process(image)
            .addOnSuccessListener { faces ->
                lastFace = faces.firstOrNull()
                lastFace?.let { face ->
                    headAngleX = face.headEulerAngleX
                    headAngleY = face.headEulerAngleY
                    headAngleZ = face.headEulerAngleZ
                    val leftEyeOpen = face.leftEyeOpenProbability ?: 0.5f
                    val rightEyeOpen = face.rightEyeOpenProbability ?: 0.5f
                    eyeOpenness = (leftEyeOpen + rightEyeOpen) / 2f
                    updateDebugInfo()
                }
                poseDetector.process(image)
                    .addOnSuccessListener { pose ->
                        lastPose = pose
                        lastPose?.let { p ->
                            val leftShoulder = p.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.LEFT_SHOULDER)
                            val rightShoulder = p.getPoseLandmark(com.google.mlkit.vision.pose.PoseLandmark.RIGHT_SHOULDER)
                            if (leftShoulder != null && rightShoulder != null) {
                                val deltaY = rightShoulder.position.y - leftShoulder.position.y
                                val deltaX = rightShoulder.position.x - leftShoulder.position.x
                                val angleRadians = kotlin.math.atan2(deltaY, deltaX)
                                shoulderAngle = (angleRadians * (180 / kotlin.math.PI)).toFloat()
                                updateDebugInfo()
                            }
                        }
                        postureMonitor.processImage(lastFace, lastPose)
                    }
                    .addOnFailureListener { e ->
                        Log.e(TAG, "포즈 감지 실패", e)
                    }
                    .addOnCompleteListener {
                        imageProxy.close()
                    }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "얼굴 감지 실패", e)
                imageProxy.close()
            }
    }

    private fun updateDebugInfo() {
        runOnUiThread {
            val debugInfo = StringBuilder()
            val headXStatus = when {
                headAngleX < -10f -> "불량"
                headAngleX > 8f -> "불량"
                headAngleX < -5f || headAngleX > 5f -> "주의"
                else -> "정상"
            }
            val headYStatus = when {
                abs(headAngleY) > 7f -> "불량"
                abs(headAngleY) > 4f -> "주의"
                else -> "정상"
            }
            val headZStatus = when {
                abs(headAngleZ) > 7f -> "불량"
                abs(headAngleZ) > 4f -> "주의"
                else -> "정상"
            }
            debugInfo.append("머리: X=${headAngleX.toInt()}° ($headXStatus) ")
            debugInfo.append("Y=${headAngleY.toInt()}° ($headYStatus) ")
            debugInfo.append("Z=${headAngleZ.toInt()}° ($headZStatus)\n")
            val normalizedShoulderAngle = when {
                shoulderAngle > 90 -> 180 - shoulderAngle
                shoulderAngle < -90 -> -180 - shoulderAngle
                else -> shoulderAngle
            }
            val shoulderStatus = when {
                abs(normalizedShoulderAngle) > 4.0f -> "불량"
                abs(normalizedShoulderAngle) > 2.0f -> "주의"
                else -> "정상"
            }
            debugInfo.append("어깨 기울기: ${shoulderAngle.toInt()}° ")
            if (abs(normalizedShoulderAngle - shoulderAngle) > 10f) {
                debugInfo.append("(${normalizedShoulderAngle.toInt()}°) ")
            }
            debugInfo.append("($shoulderStatus)\n")
            debugInfo.append("눈 열림: ${(eyeOpenness * 100).toInt()}%\n")
            debugInfo.append("자세: $currentPostureState")
            debugTextView.text = debugInfo.toString()

            if (shoulderStatus == "불량") {
                addPostureIssue(PostureMonitor.PostureType.SHOULDER_TILT, "어깨가 기울어짐")
            } else {
                removePostureIssue(PostureMonitor.PostureType.SHOULDER_TILT)
            }
            if (headXStatus == "불량" && headAngleX < 0) {
                addPostureIssue(PostureMonitor.PostureType.HEAD_FORWARD_TILT, "머리가 앞으로 기울어짐")
            } else {
                removePostureIssue(PostureMonitor.PostureType.HEAD_FORWARD_TILT)
            }
            if (headXStatus == "불량" && headAngleX > 0) {
                addPostureIssue(PostureMonitor.PostureType.HEAD_BACKWARD_TILT, "머리가 뒤로 기울어짐")
            } else {
                removePostureIssue(PostureMonitor.PostureType.HEAD_BACKWARD_TILT)
            }
            if (headYStatus == "불량") {
                addPostureIssue(PostureMonitor.PostureType.HEAD_SIDE_TILT, "머리가 옆으로 기울어짐")
            } else {
                removePostureIssue(PostureMonitor.PostureType.HEAD_SIDE_TILT)
            }
            if (headZStatus == "불량") {
                addPostureIssue(PostureMonitor.PostureType.HEAD_ROLL_TILT, "머리가 회전됨")
            } else {
                removePostureIssue(PostureMonitor.PostureType.HEAD_ROLL_TILT)
            }
            if (eyeOpenness < 0.3f) {
                addPostureIssue(PostureMonitor.PostureType.DROWSINESS, "졸음 감지")
            } else {
                removePostureIssue(PostureMonitor.PostureType.DROWSINESS)
            }
        }
    }

    private fun switchCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
            CameraSelector.LENS_FACING_BACK
        else
            CameraSelector.LENS_FACING_FRONT
        bindCameraUseCases()
    }

    private fun connectBluetooth() {
        if (!bluetoothManager.isBluetoothAvailable()) {
            bluetoothManager.requestBluetoothEnable(this)
            return
        }
        val currentState = bluetoothManager.connectionState.value
        when (currentState) {
            BluetoothManager.ConnectionState.CONNECTING -> {
                showToast("이미 연결 중입니다...")
            }
            BluetoothManager.ConnectionState.CONNECTED -> {
                showToast("이미 연결되어 있습니다. 재연결을 시도합니다.")
                bluetoothManager.closeConnection()
                Handler(Looper.getMainLooper()).postDelayed({
                    bluetoothManager.initializeBluetooth()
                }, 500)
            }
            else -> {
                bluetoothManager.closeConnection()
                Handler(Looper.getMainLooper()).postDelayed({
                    bluetoothManager.initializeBluetooth()
                    showToast("독서대에 연결 중...")
                }, 300)
            }
        }
    }

    // PostureMonitor.PostureListener 구현
    override fun onStateChanged(state: PostureMonitor.MonitorState) {
        Log.d(TAG, "상태 변경: $state")
        runOnUiThread {
            updateStatusUI(state)
        }
    }

    override fun onDebugInfo(debugInfo: String) {
        runOnUiThread {
            // PostureMonitor의 로그 메시지를 통해 자리비움이나 자세 회복 이벤트 처리
            if (debugInfo.contains("자리비움 감지") && debugInfo.contains("문제로 확정")) {
                addPostureIssue(PostureMonitor.PostureType.USER_AWAY, "자리비움")
            }
            else if (debugInfo.contains("자세 회복") || debugInfo.contains("상태에서 회복")) {
                if (debugInfo.contains("머리 앞으로 기울어짐") && debugInfo.contains("회복")) {
                    removePostureIssue(PostureMonitor.PostureType.HEAD_FORWARD_TILT)
                }
                else if (debugInfo.contains("머리 뒤로 기울어짐") && debugInfo.contains("회복")) {
                    removePostureIssue(PostureMonitor.PostureType.HEAD_BACKWARD_TILT)
                }
                else if (debugInfo.contains("머리 옆으로 기울어짐") && debugInfo.contains("회복")) {
                    removePostureIssue(PostureMonitor.PostureType.HEAD_SIDE_TILT)
                }
                else if (debugInfo.contains("머리 회전") && debugInfo.contains("회복")) {
                    removePostureIssue(PostureMonitor.PostureType.HEAD_ROLL_TILT)
                }
                else if (debugInfo.contains("어깨 기울기") && debugInfo.contains("회복")) {
                    removePostureIssue(PostureMonitor.PostureType.SHOULDER_TILT)
                }
                else if (debugInfo.contains("졸음") && debugInfo.contains("회복")) {
                    removePostureIssue(PostureMonitor.PostureType.DROWSINESS)
                }
                else if (debugInfo.contains("자리비움") && debugInfo.contains("회복")) {
                    removePostureIssue(PostureMonitor.PostureType.USER_AWAY)
                }
            }
            // 디버그 정보 업데이트는 변화가 있을 때만 수행
            if (debugInfo.contains("감지") || debugInfo.contains("회복") || debugInfo.contains("확정")) {
                updateDebugInfo()
            }
        }
    }

    private fun updateStatusUI(state: PostureMonitor.MonitorState) {
        statusTextView.text = if (currentPostureType == PostureMonitor.PostureType.NORMAL)
            "자세가 바릅니다" else currentPostureState
        Log.d(TAG, "UI 상태 업데이트: ${statusTextView.text}")
    }

    private fun updateBluetoothStatusUI(state: BluetoothManager.ConnectionState) {
        val statusText = when (state) {
            BluetoothManager.ConnectionState.DISCONNECTED -> "연결 끊김"
            BluetoothManager.ConnectionState.CONNECTING -> "연결 중..."
            BluetoothManager.ConnectionState.CONNECTED -> "연결됨"
            BluetoothManager.ConnectionState.CONNECTION_FAILED -> "연결 실패"
            BluetoothManager.ConnectionState.PERMISSION_DENIED -> "권한 거부됨"
            else -> "알 수 없음"
        }
        btStatusTextView.text = "독서대: $statusText"
        val textColor = when (state) {
            BluetoothManager.ConnectionState.CONNECTED -> android.graphics.Color.GREEN
            BluetoothManager.ConnectionState.CONNECTING -> android.graphics.Color.BLUE
            BluetoothManager.ConnectionState.DISCONNECTED,
            BluetoothManager.ConnectionState.CONNECTION_FAILED,
            BluetoothManager.ConnectionState.PERMISSION_DENIED -> ContextCompat.getColor(this, R.color.error_red)
            else -> ContextCompat.getColor(this, R.color.error_red)
        }
        btStatusTextView.setTextColor(textColor)
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun allPermissionsGranted() = getRequiredPermissions().all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                showToast("카메라 및 블루투스 권한이 필요합니다.")
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        postureMonitor.release()
        bluetoothManager.releaseBluetooth()
    }
}