package com.example.rightpose

import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.util.UUID

/**
 * 블루투스 관리 클래스
 * 자세 모니터링 장치(ESP32)와의 블루투스 통신을 담당
 */
object BluetoothManager {
    private const val TAG = "BluetoothManager"

    // 블루투스 요청 코드
    private const val REQUEST_ENABLE_BT = 101

    // ESP32 장치 이름 - ESP32 코드의 deviceName과 일치해야 함
    private const val DEVICE_NAME = "PoseDetect-BLE"

    // SPP (Serial Port Profile) UUID
    private val SPP_UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")

    // 블루투스 어댑터
    private var bluetoothAdapter: BluetoothAdapter? = null

    // 블루투스 소켓 및 통신 스트림
    private var bluetoothSocket: BluetoothSocket? = null
    private var inputStream: InputStream? = null
    private var outputStream: OutputStream? = null

    // 연결 상태
    private val _connectionState = MutableLiveData<ConnectionState>()
    val connectionState: LiveData<ConnectionState> = _connectionState

    // 연결 상태 열거형
    enum class ConnectionState {
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        CONNECTION_FAILED,
        PERMISSION_DENIED
    }

    // 송신한 명령 기록
    private val commandHistory = mutableListOf<String>()

    // 리스너 인터페이스
    interface BluetoothManagerListener {
        // 이 인터페이스를 구현하는 클래스에서 처리할 메서드
    }

    private var listener: BluetoothManagerListener? = null

    // 명령 상수
    const val CMD_NORMAL = "NORMAL"
    const val CMD_POSTURE_ALERT = "POSTURE_ALERT"
    const val CMD_POSTURE_ALERT_L1 = "POSTURE_ALERT_L1"
    const val CMD_POSTURE_ALERT_L3 = "POSTURE_ALERT_L3"
    const val CMD_POSTURE_ALERT_L5 = "POSTURE_ALERT_L5"
    const val CMD_DROWSINESS_ALERT = "DROWSINESS_ALERT"
    const val CMD_FOCUS_ALERT = "FOCUS_ALERT"

    // 부재 감지 관련 명령 추가
    const val CMD_USER_AWAY = "USER_AWAY"
    const val CMD_USER_AWAY_L1 = "USER_AWAY_L1"
    const val CMD_USER_AWAY_L3 = "USER_AWAY_L3"
    const val CMD_USER_AWAY_L5 = "USER_AWAY_L5"
    const val CMD_USER_RETURN = "USER_RETURN"

    // 자리 비움 지속 시간에 따른 알림 레벨
    private const val USER_AWAY_LEVEL1_SECONDS = 30  // 30초 후 1단계 알림
    private const val USER_AWAY_LEVEL3_SECONDS = 120 // 2분 후 3단계 알림
    private const val USER_AWAY_LEVEL5_SECONDS = 300 // 5분 후 5단계 알림

    // 자리 비움 관련 변수
    private var userAwayStartTime: Long = 0
    private var currentAwayLevel = 0
    private var awayMonitoringHandler: Handler? = null
    private val awayMonitoringRunnable = object : Runnable {
        override fun run() {
            if (userAwayStartTime > 0) {
                val awayDuration = (System.currentTimeMillis() - userAwayStartTime) / 1000

                // 시간 경과에 따라 알림 레벨 증가
                val newLevel = when {
                    awayDuration >= USER_AWAY_LEVEL5_SECONDS -> 5
                    awayDuration >= USER_AWAY_LEVEL3_SECONDS -> 3
                    awayDuration >= USER_AWAY_LEVEL1_SECONDS -> 1
                    else -> 0
                }

                // 레벨이 변경된 경우에만 명령 전송
                if (newLevel != currentAwayLevel) {
                    currentAwayLevel = newLevel

                    // 레벨에 따른 명령 전송
                    val command = when (currentAwayLevel) {
                        1 -> CMD_USER_AWAY_L1
                        3 -> CMD_USER_AWAY_L3
                        5 -> CMD_USER_AWAY_L5
                        else -> CMD_USER_AWAY
                    }

                    sendCommand(command)
                    Log.d(TAG, "자리 비움 경과 시간: ${awayDuration}초, 알림 레벨: $currentAwayLevel")
                }

                // 다음 확인을 위한 핸들러 예약 (10초마다)
                awayMonitoringHandler?.postDelayed(this, 10000)
            }
        }
    }

    init {
        _connectionState.value = ConnectionState.DISCONNECTED
        awayMonitoringHandler = Handler(Looper.getMainLooper())
    }

    /**
     * 블루투스 리스너 설정
     */
    fun setBluetoothListener(listener: BluetoothManagerListener) {
        this.listener = listener
    }

    /**
     * 블루투스 초기화
     */
    fun initializeBluetooth() {
        Log.d(TAG, "블루투스 초기화")

        bluetoothAdapter = BluetoothAdapter.getDefaultAdapter()

        if (bluetoothAdapter == null) {
            Log.e(TAG, "블루투스를 지원하지 않는 기기")
            return
        }

        if (!bluetoothAdapter!!.isEnabled) {
            Log.w(TAG, "블루투스가 비활성화 상태")
            return
        }

        _connectionState.postValue(ConnectionState.CONNECTING)

        // 별도 스레드에서 블루투스 연결 시도
        Thread {
            try {
                connectToDevice()
            } catch (e: Exception) {
                Log.e(TAG, "블루투스 연결 실패", e)
                _connectionState.postValue(ConnectionState.CONNECTION_FAILED)
            }
        }.start()
    }

    /**
     * 블루투스 장치에 연결
     */
    @Throws(IOException::class)
    private fun connectToDevice() {
        Log.d(TAG, "블루투스 장치 검색 및 연결 시도")

        val pairedDevices = bluetoothAdapter?.bondedDevices
        var targetDevice: BluetoothDevice? = null

        if (pairedDevices != null) {
            for (device in pairedDevices) {
                Log.d(TAG, "페어링된 장치: ${device.name}")
                if (device.name == DEVICE_NAME) {
                    targetDevice = device
                    Log.d(TAG, "대상 장치 발견: $DEVICE_NAME")
                    break
                }
            }
        }

        if (targetDevice == null) {
            Log.e(TAG, "대상 장치를 찾을 수 없음: $DEVICE_NAME")
            _connectionState.postValue(ConnectionState.CONNECTION_FAILED)
            return
        }

        try {
            // 기존 연결이 있으면 닫기
            closeConnection()

            // 새 연결 생성
            bluetoothSocket = targetDevice.createRfcommSocketToServiceRecord(SPP_UUID)
            bluetoothSocket?.connect()

            // 입출력 스트림 설정
            inputStream = bluetoothSocket?.inputStream
            outputStream = bluetoothSocket?.outputStream

            Log.d(TAG, "블루투스 연결 성공: $DEVICE_NAME")
            _connectionState.postValue(ConnectionState.CONNECTED)

            // 별도 스레드에서 수신 데이터 모니터링
            startReceiving()

            // 연결 후 keepalive 명령 전송
            sendKeepAliveCommand()
        } catch (e: IOException) {
            Log.e(TAG, "블루투스 소켓 연결 실패", e)
            closeConnection()
            _connectionState.postValue(ConnectionState.CONNECTION_FAILED)
            throw e
        }
    }

    /**
     * 블루투스 연결 종료
     */
    fun closeConnection() {
        try {
            inputStream?.close()
            outputStream?.close()
            bluetoothSocket?.close()
        } catch (e: IOException) {
            Log.e(TAG, "연결 종료 중 오류", e)
        } finally {
            inputStream = null
            outputStream = null
            bluetoothSocket = null
            if (_connectionState.value == ConnectionState.CONNECTED) {
                _connectionState.postValue(ConnectionState.DISCONNECTED)
            }
            stopAwayMonitoring() // 연결 종료 시 자리 비움 모니터링 중지
        }
    }

    /**
     * 블루투스 리소스 해제
     */
    fun releaseBluetooth() {
        closeConnection()
        awayMonitoringHandler?.removeCallbacksAndMessages(null)
        awayMonitoringHandler = null
    }

    /**
     * 명령 전송
     */
    fun sendCommand(command: String) {
        if (_connectionState.value != ConnectionState.CONNECTED) {
            Log.w(TAG, "명령 전송 실패: 연결되지 않음")
            return
        }

        Thread {
            try {
                val jsonCommand = "{\"type\":\"$command\"}\n"
                outputStream?.write(jsonCommand.toByteArray())
                Log.d(TAG, "명령 전송: $command")
                commandHistory.add(command)

                // 명령 이력이 너무 길어지면 오래된 것 제거
                if (commandHistory.size > 20) {
                    commandHistory.removeAt(0)
                }
            } catch (e: IOException) {
                Log.e(TAG, "명령 전송 실패", e)
                // 연결이 끊어진 것으로 간주
                Handler(Looper.getMainLooper()).post {
                    _connectionState.value = ConnectionState.CONNECTION_FAILED
                }
            }
        }.start()
    }

    /**
     * Keepalive 명령 전송
     */
    private fun sendKeepAliveCommand() {
        sendCommand("KEEPALIVE")

        // 주기적으로 Keepalive 전송
        Handler(Looper.getMainLooper()).postDelayed({
            if (_connectionState.value == ConnectionState.CONNECTED) {
                sendKeepAliveCommand()
            }
        }, 30000) // 30초 간격
    }

    /**
     * 블루투스 수신 시작
     */
    private fun startReceiving() {
        Thread {
            val buffer = ByteArray(1024)
            var bytes: Int

            while (_connectionState.value == ConnectionState.CONNECTED) {
                try {
                    bytes = inputStream?.read(buffer) ?: -1

                    if (bytes > 0) {
                        val received = String(buffer, 0, bytes)
                        Log.d(TAG, "데이터 수신: $received")

                        // 필요한 경우 수신 데이터 처리
                    }
                } catch (e: IOException) {
                    Log.e(TAG, "데이터 수신 오류", e)
                    break
                }
            }

            // 연결 종료
            Handler(Looper.getMainLooper()).post {
                if (_connectionState.value == ConnectionState.CONNECTED) {
                    _connectionState.value = ConnectionState.DISCONNECTED
                }
            }
        }.start()
    }

    /**
     * 자리 비움 모니터링 시작
     * MainActivity에서 사용자 부재가 감지되었을 때 호출됨
     */
    fun startAwayMonitoring() {
        userAwayStartTime = System.currentTimeMillis()
        currentAwayLevel = 0

        // 초기 자리 비움 명령 전송
        sendCommand(CMD_USER_AWAY)
        Log.d(TAG, "자리 비움 모니터링 시작")

        // 모니터링 핸들러 시작
        awayMonitoringHandler?.removeCallbacks(awayMonitoringRunnable)
        awayMonitoringHandler?.postDelayed(awayMonitoringRunnable, 10000) // 10초 후 첫 체크
    }

    /**
     * 자리 비움 모니터링 중지
     * MainActivity에서 사용자가 돌아온 것이 감지되었을 때 호출됨
     */
    fun stopAwayMonitoring() {
        if (userAwayStartTime > 0) {
            userAwayStartTime = 0
            currentAwayLevel = 0
            awayMonitoringHandler?.removeCallbacks(awayMonitoringRunnable)

            // 사용자가 돌아왔음을 알리는 명령 전송
            sendCommand(CMD_USER_RETURN)
            Log.d(TAG, "자리 비움 모니터링 중지 - 사용자 복귀")
        }
    }

    /**
     * 블루투스 사용 가능 여부 확인
     */
    fun isBluetoothAvailable(): Boolean {
        return bluetoothAdapter != null && bluetoothAdapter!!.isEnabled
    }

    /**
     * 연결 상태 확인
     */
    fun isConnected(): Boolean {
        return _connectionState.value == ConnectionState.CONNECTED
    }

    /**
     * 블루투스 활성화 요청
     */
    fun requestBluetoothEnable(activity: Activity) {
        bluetoothAdapter?.let {
            if (!it.isEnabled) {
                val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
                activity.startActivityForResult(enableBtIntent, REQUEST_ENABLE_BT)
            }
        }
    }

    /**
     * 블루투스 권한 확인
     */
    fun checkPermissions(context: Context): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val hasConnectPermission = context.checkSelfPermission(android.Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED
            val hasScanPermission = context.checkSelfPermission(android.Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED

            if (!hasConnectPermission || !hasScanPermission) {
                _connectionState.value = ConnectionState.PERMISSION_DENIED
                return false
            }
        } else {
            val hasBluetoothPermission = context.checkSelfPermission(android.Manifest.permission.BLUETOOTH) == PackageManager.PERMISSION_GRANTED
            val hasAdminPermission = context.checkSelfPermission(android.Manifest.permission.BLUETOOTH_ADMIN) == PackageManager.PERMISSION_GRANTED

            if (!hasBluetoothPermission || !hasAdminPermission) {
                _connectionState.value = ConnectionState.PERMISSION_DENIED
                return false
            }
        }

        return true
    }

    /**
     * 명령 이력 반환
     */
    fun getCommandHistory(): List<String> {
        return commandHistory.toList()
    }
}