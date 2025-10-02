package com.aifit.app

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.aifit.app/pytorch"
    // PyTorch 제거: TFLite로 이전. 네이티브 추론은 사용하지 않음.

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "loadModel" -> {
                    // PyTorch 네이티브 로드는 더 이상 지원하지 않음. Dart에서 TFLite 사용.
                    result.error("NOT_SUPPORTED", "PyTorch 네이티브 로드는 비활성화됨. Dart에서 TFLite 사용", null)
                }
                "runInference" -> {
                    // 네이티브 PyTorch 추론은 더 이상 지원하지 않음. Dart TFLite 경로 사용.
                    result.error("NOT_SUPPORTED", "네이티브 PyTorch 추론 비활성화. Dart에서 TFLite로 추론", null)
                }
                "getMemoryInfo" -> {
                    try {
                        val memoryInfo = getMemoryInfo()
                        result.success(memoryInfo)
                    } catch (e: Exception) {
                        result.error("MEMORY_INFO_ERROR", "메모리 정보 가져오기 실패: ${e.message}", e.toString())
                    }
                }
                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    /**
     * Copies an asset to the app's file system and returns its absolute path.
     */
    @Throws(IOException::class)
    private fun assetFilePath(assetKey: String, destinationFile: String): String {
        val file = File(filesDir, destinationFile)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetKey).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
            return file.absolutePath
        }
    }

    /**
     * 현재 메모리 사용 정보 반환
     */
    private fun getMemoryInfo(): Map<String, Any> {
        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)

        // Native heap 메모리 정보
        val nativeHeapSize = Debug.getNativeHeapSize() / (1024 * 1024) // MB
        val nativeHeapAllocated = Debug.getNativeHeapAllocatedSize() / (1024 * 1024) // MB
        val nativeHeapFree = Debug.getNativeHeapFreeSize() / (1024 * 1024) // MB

        return mapOf(
            "totalMemory" to (memoryInfo.totalMem / (1024 * 1024)), // MB
            "availableMemory" to (memoryInfo.availMem / (1024 * 1024)), // MB
            "usedMemory" to ((memoryInfo.totalMem - memoryInfo.availMem) / (1024 * 1024)), // MB
            "threshold" to (memoryInfo.threshold / (1024 * 1024)), // MB
            "lowMemory" to memoryInfo.lowMemory,
            "nativeHeapSize" to nativeHeapSize,
            "nativeHeapAllocated" to nativeHeapAllocated,
            "nativeHeapFree" to nativeHeapFree
        )
    }

    /**
     * 메모리 사용량을 로그로 출력
     */
    private fun logMemoryUsage(label: String) {
        val memoryInfo = getMemoryInfo()
        android.util.Log.d("PyTorch", "💾 [$label] Android 메모리:")
        android.util.Log.d("PyTorch", "  - 총 메모리: ${memoryInfo["totalMemory"]} MB")
        android.util.Log.d("PyTorch", "  - 사용 가능 메모리: ${memoryInfo["availableMemory"]} MB")
        android.util.Log.d("PyTorch", "  - 사용 중 메모리: ${memoryInfo["usedMemory"]} MB")
        android.util.Log.d("PyTorch", "  - Native Heap 크기: ${memoryInfo["nativeHeapSize"]} MB")
        android.util.Log.d("PyTorch", "  - Native Heap 할당: ${memoryInfo["nativeHeapAllocated"]} MB")
        android.util.Log.d("PyTorch", "  - Native Heap 여유: ${memoryInfo["nativeHeapFree"]} MB")
        android.util.Log.d("PyTorch", "  - 메모리 부족 상태: ${memoryInfo["lowMemory"]}")
    }
}
