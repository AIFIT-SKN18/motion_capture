package com.example.aifit

import android.content.Context
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.example.aifit/tflite"
    private var interpreter: Interpreter? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "loadModel" -> {
                    try {
                        val modelPath = call.argument<String>("modelPath")
                        if (modelPath != null) {
                            loadModel(modelPath)
                            result.success("Model loaded successfully")
                        } else {
                            result.error("INVALID_ARGUMENT", "Model path is null", null)
                        }
                    } catch (e: Exception) {
                        result.error("LOAD_ERROR", "Failed to load model: ${e.message}", null)
                    }
                }
                "runInference" -> {
                    try {
                        val inputData = call.argument<List<Double>>("inputData")
                        val shape = call.argument<List<Int>>("shape")
                        
                        if (inputData != null && shape != null) {
                            val output = runInference(inputData, shape)
                            result.success(output)
                        } else {
                            result.error("INVALID_ARGUMENT", "Input data or shape is null", null)
                        }
                    } catch (e: Exception) {
                        result.error("INFERENCE_ERROR", "Failed to run inference: ${e.message}", null)
                    }
                }
                "closeModel" -> {
                    try {
                        interpreter?.close()
                        interpreter = null
                        result.success("Model closed")
                    } catch (e: Exception) {
                        result.error("CLOSE_ERROR", "Failed to close model: ${e.message}", null)
                    }
                }
                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    private fun loadModel(modelPath: String) {
        val modelFile = loadModelFile(assets, modelPath)
        
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            setUseNNAPI(false)
            // Enable XNNPACK delegate for better performance/stability on CPU
            setUseXNNPACK(true)
        }
        
        interpreter = Interpreter(modelFile, options)

        // Debug: print tensor dtypes and shapes
        try {
            val inputTensor = interpreter!!.getInputTensor(0)
            val outputTensor = interpreter!!.getOutputTensor(0)
            android.util.Log.i("TFLite", "Input tensor type: ${inputTensor.dataType()}, shape: ${inputTensor.shape().contentToString()}")
            android.util.Log.i("TFLite", "Output tensor type: ${outputTensor.dataType()}, shape: ${outputTensor.shape().contentToString()}")
        } catch (_: Exception) {}
    }

    private fun loadModelFile(context: android.content.res.AssetManager, modelPath: String): MappedByteBuffer {
        try {
            val fileDescriptor = context.openFd(modelPath)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        } catch (e: Exception) {
            throw Exception("Failed to load model file at path: $modelPath. Error: ${e.message}")
        }
    }

    private fun runInference(inputData: List<Double>, shape: List<Int>): List<Float> {
        if (interpreter == null) {
            throw Exception("Model not loaded")
        }

        // 모델의 실제 입력 텐서 shape 조회
        val tensor = interpreter!!.getInputTensor(0)
        val tensorShape = tensor.shape() // e.g., [1,64,25,1,3] or [1,3,64,25,1]
        val dims = tensorShape.map { it }
        val inputSize = dims.reduce { acc, i -> acc * i }

        val inputBuffer = ByteBuffer.allocateDirect(inputSize * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        // 입력 데이터 재배열: 앱은 기본적으로 (T,V,M,C) 순서로 전달
        // 텐서가 (1,64,25,1,3)면 그대로 순차 복사, (1,3,64,25,1)이면 (T,V,M,C)->(C,T,V,M)로 재배열
        val isTVMC = (dims.size == 5 && dims[1] == 64 && dims[4] == 3)
        val T = 64; val V = 25; val M = 1; val C = 3

        if (isTVMC) {
            // 그대로 복사
            for (v in inputData) inputBuffer.putFloat(v.toFloat())
        } else {
            // 대상이 (1,3,64,25,1)인 경우로 가정 → (C,T,V,M)
            // 소스는 (T,V,M,C)
            for (c in 0 until C) {
                for (t in 0 until T) {
                    for (v in 0 until V) {
                        for (m in 0 until M) {
                            val src = (((t * V) + v) * M + m) * C + c
                            inputBuffer.putFloat(inputData[src].toFloat())
                        }
                    }
                }
            }
        }
        inputBuffer.rewind()

        // 출력 버퍼 생성 [1, 6]
        val outputBuffer = ByteBuffer.allocateDirect(6 * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        // 추론 실행
        interpreter?.run(inputBuffer, outputBuffer)
        
        // 출력 데이터 추출
        outputBuffer.rewind()
        val output = mutableListOf<Float>()
        for (i in 0 until 6) {
            output.add(outputBuffer.float)
        }

        return output
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter?.close()
    }
}
