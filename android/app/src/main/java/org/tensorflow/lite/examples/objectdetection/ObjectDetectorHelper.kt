package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.util.*

class ObjectDetectorHelper(
        var threshold: Float = 0.5f,
        var numThreads: Int = 2,
        var maxResults: Int = 3,
        var currentDelegate: Int = 0,
        var currentModel: Int = 0,
        val context: Context,
        val objectDetectorListener: DetectorListener?
) {

    private var objectDetector: ObjectDetector? = null
    private var textToSpeech: TextToSpeech? = null
    private var lastAnnouncementTime = 0L

    init {
        setupObjectDetector()
        textToSpeech = TextToSpeech(context) { status ->
            if (status != TextToSpeech.ERROR) {
                textToSpeech?.language = Locale.getDefault()
            } else {
                Log.e("ObjectDetectorHelper", "TextToSpeech initialization failed")
            }
        }
    }

    fun clearObjectDetector() {
        objectDetector = null
    }

    fun setupObjectDetector() {
        val optionsBuilder =
                ObjectDetector.ObjectDetectorOptions.builder()
                        .setScoreThreshold(threshold)
                        .setMaxResults(maxResults)

        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        when (currentDelegate) {
            DELEGATE_CPU -> {
            }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    objectDetectorListener?.onError("GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName =
                when (currentModel) {
                    MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
                    MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
                    MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
                    MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
                    else -> "mobilenetv1.tflite"
                }

        try {
            objectDetector =
                    ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            objectDetectorListener?.onError(
                    "Object detector failed to initialize. See error logs for details"
            )
            Log.e("ObjectDetectorHelper", "TFLite failed to load model with error: " + e.message)
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null) {
            setupObjectDetector()
        }

        var inferenceTime = SystemClock.uptimeMillis()

        val imageProcessor =
                ImageProcessor.Builder()
                        .add(Rot90Op(-imageRotation / 90))
                        .build()

        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val results = objectDetector?.detect(tensorImage)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (results != null) {
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastAnnouncementTime >= 5000) {
                var closestDistance = Float.MAX_VALUE
                var closestLabel: String? = null
                var closestCenterX = 0f
                var closestCenterY = 0f
                for (result in results) {
                    for (category in result.categories) {
                        if (category.score > threshold) {
                            val centerX = (result.boundingBox.left + result.boundingBox.right) / 2f
                            val centerY = (result.boundingBox.top + result.boundingBox.bottom) / 2f
                            val distance = calculateDistanceToCamera(centerX, centerY, image.width, image.height)
                            if (distance < closestDistance) {
                                closestDistance = distance
                                closestLabel = when (category.label) {
                                    "person" -> "insan"
                                    "cell phone" -> "cep telefonu"
                                    else -> category.label
                                }
                                closestCenterX = centerX
                                closestCenterY = centerY
                            }
                        }
                    }
                }
                if (closestLabel != null) {
                    //textToSpeech?.speak(closestLabel, TextToSpeech.QUEUE_ADD, null, null)
                    val distanceInCm = (closestDistance / 10).toInt() // Assuming 1 pixel is 1/10 cm
                    val distanceMessage = "$closestLabel, $distanceInCm santimetre"
                    textToSpeech?.speak(distanceMessage, TextToSpeech.QUEUE_ADD, null, null)
                }
                lastAnnouncementTime = currentTime
            }
        }

        objectDetectorListener?.onResults(
                results,
                inferenceTime,
                tensorImage.height,
                tensorImage.width
        )
    }

    private fun calculateDistanceToCamera(centerX: Float, centerY: Float, imageWidth: Int, imageHeight: Int): Float {
        val imageCenterX = imageWidth / 2f
        val imageCenterY = imageHeight / 2f
        val pixelDistance = Math.sqrt(((centerX - imageCenterX) * (centerX - imageCenterX) + (centerY - imageCenterY) * (centerY - imageCenterY)).toDouble())
        return pixelDistance.toFloat() // 1 pixel 1 milimetre olsun
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
                results: MutableList<Detection>?,
                inferenceTime: Long,
                imageHeight: Int,
                imageWidth: Int
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
    }
}
