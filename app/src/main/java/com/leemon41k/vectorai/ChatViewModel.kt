package com.leemon41k.vectorai

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class ChatMessage(
    val isUser: Boolean,
    val text: String
)

data class ChatUiState(
    val modelReady: Boolean = false,
    val modelExists: Boolean = false,
    val isDownloading: Boolean = false,
    val isLoading: Boolean = false,
    val isBusy: Boolean = false,
    val downloadProgress: Float = 0f,
    val downloadedBytes: Long = 0L,
    val totalBytes: Long = 0L,
    val error: String? = null
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    companion object {
        private const val MODEL_URL =
            "https://huggingface.co/litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B.litertlm"
        private const val MODEL_FILE_NAME = "vectorai-qwen3-0.6b.litertlm"
        private const val MIN_MODEL_SIZE = 500_000_000L
        private const val MAX_MODEL_SIZE = 700_000_000L
    }

    private val executor = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(SupervisorJob() + executor.asCoroutineDispatcher())
    private val mutex = Mutex()
    private val modelFile = File(application.filesDir, MODEL_FILE_NAME)

    private var engine: Engine? = null
    private var conversation: Conversation? = null

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    fun prepare() {
        scope.launch {
            mutex.withLock {
                if (_state.value.modelReady || _state.value.isLoading || _state.value.isDownloading) {
                    return@withLock
                }
                if (modelFile.isFile && modelFile.length() in MIN_MODEL_SIZE..MAX_MODEL_SIZE) {
                    _state.update { it.copy(modelExists = true, error = null) }
                    loadModelLocked()
                } else {
                    modelFile.delete()
                    _state.value = ChatUiState()
                }
            }
        }
    }

    fun downloadModel() {
        scope.launch {
            mutex.withLock {
                if (_state.value.isDownloading || _state.value.isLoading || _state.value.modelReady) {
                    return@withLock
                }
                downloadAndLoadLocked()
            }
        }
    }

    fun redownloadModel() {
        scope.launch {
            mutex.withLock {
                releaseLocked()
                modelFile.delete()
                downloadAndLoadLocked()
            }
        }
    }

    fun retryLoadingModel() {
        scope.launch {
            mutex.withLock {
                if (modelFile.isFile && modelFile.length() in MIN_MODEL_SIZE..MAX_MODEL_SIZE) {
                    loadModelLocked()
                } else {
                    downloadAndLoadLocked()
                }
            }
        }
    }

    fun send(text: String) {
        val cleaned = text.trim()
        if (cleaned.isEmpty()) return

        scope.launch {
            if (!_state.value.modelReady || _state.value.isBusy) return@launch

            _messages.update { it + ChatMessage(isUser = true, text = cleaned) + ChatMessage(isUser = false, text = "") }
            _state.update { it.copy(isBusy = true, error = null) }

            val answer = StringBuilder()
            try {
                val activeConversation = conversation ?: throw IllegalStateException("Локальная модель ещё не готова.")
                activeConversation.sendMessageAsync(cleaned).collect { chunk ->
                    answer.append(chunk.toString())
                    updateLastAssistant(answer.toString())
                }
                if (answer.isEmpty()) {
                    updateLastAssistant("Не удалось получить ответ. Попробуй сформулировать вопрос иначе.")
                }
            } catch (error: Exception) {
                val message = error.localizedMessage?.takeIf { it.isNotBlank() }
                    ?: "Неизвестная ошибка"
                updateLastAssistant("Не удалось получить ответ: $message")
            } finally {
                _state.update { it.copy(isBusy = false) }
            }
        }
    }

    private fun updateLastAssistant(text: String) {
        _messages.update { current ->
            if (current.isNotEmpty() && !current.last().isUser) {
                current.dropLast(1) + current.last().copy(text = text)
            } else {
                current + ChatMessage(isUser = false, text = text)
            }
        }
    }

    private suspend fun downloadAndLoadLocked() {
        val partialFile = File(getApplication<Application>().cacheDir, "$MODEL_FILE_NAME.part")
        _state.value = ChatUiState(isDownloading = true, error = null)

        try {
            withContext(Dispatchers.IO) {
                val connection = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 20_000
                    readTimeout = 60_000
                    instanceFollowRedirects = true
                    setRequestProperty("User-Agent", "VectorAI-Android")
                    setRequestProperty("Accept-Encoding", "identity")
                }

                try {
                    val responseCode = connection.responseCode
                    if (responseCode !in 200..299) {
                        throw IOException("Сервер модели ответил кодом $responseCode.")
                    }

                    val expectedSize = connection.contentLengthLong
                    var downloaded = 0L
                    connection.inputStream.use { input ->
                        FileOutputStream(partialFile).use { output ->
                            val buffer = ByteArray(1024 * 1024)
                            while (true) {
                                val count = input.read(buffer)
                                if (count < 0) break
                                output.write(buffer, 0, count)
                                downloaded += count
                                val progress = if (expectedSize > 0L) {
                                    (downloaded.toDouble() / expectedSize.toDouble()).toFloat().coerceIn(0f, 1f)
                                } else {
                                    0f
                                }
                                _state.update {
                                    it.copy(
                                        downloadProgress = progress,
                                        downloadedBytes = downloaded,
                                        totalBytes = expectedSize
                                    )
                                }
                            }
                            output.fd.sync()
                        }
                    }

                    if (expectedSize > 0L && partialFile.length() != expectedSize) {
                        throw IOException("Загрузка прервалась: размер файла не совпадает.")
                    }
                    if (partialFile.length() !in MIN_MODEL_SIZE..MAX_MODEL_SIZE) {
                        throw IOException("Размер загруженной модели не совпадает с ожидаемым.")
                    }

                    if (modelFile.exists() && !modelFile.delete()) {
                        throw IOException("Не удалось заменить старый файл модели.")
                    }
                    if (!partialFile.renameTo(modelFile)) {
                        throw IOException("Не удалось сохранить файл модели.")
                    }
                } finally {
                    connection.disconnect()
                }
            }

            _state.update { it.copy(isDownloading = false, modelExists = true, isLoading = true) }
            loadModelLocked()
        } catch (error: Exception) {
            partialFile.delete()
            _state.update {
                it.copy(
                    isDownloading = false,
                    isLoading = false,
                    modelReady = false,
                    modelExists = modelFile.exists(),
                    error = error.localizedMessage ?: "Не удалось загрузить модель."
                )
            }
        }
    }

    private suspend fun loadModelLocked() {
        _state.update { it.copy(isLoading = true, modelExists = true, error = null) }

        try {
            val loadedEngine = try {
                createInitializedEngine(Backend.GPU())
            } catch (_: Exception) {
                createInitializedEngine(Backend.CPU())
            }

            engine = loadedEngine
            conversation = loadedEngine.createConversation(
                ConversationConfig(
                    systemInstruction = Contents.of(VECTORAI_SYSTEM_PROMPT),
                    samplerConfig = SamplerConfig(
                        topK = 20,
                        topP = 0.9,
                        temperature = 0.7
                    )
                )
            )

            _state.update {
                it.copy(
                    modelReady = true,
                    modelExists = true,
                    isLoading = false,
                    isDownloading = false,
                    error = null
                )
            }
        } catch (error: Exception) {
            closeResources()
            _state.update {
                it.copy(
                    modelReady = false,
                    modelExists = modelFile.exists(),
                    isLoading = false,
                    isDownloading = false,
                    error = error.localizedMessage ?: "Не удалось запустить локальную модель."
                )
            }
        }
    }

    private fun createInitializedEngine(backend: Backend): Engine {
        val candidate = Engine(
            EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = backend,
                cacheDir = getApplication<Application>().cacheDir.absolutePath
            )
        )
        candidate.initialize()
        return candidate
    }

    private suspend fun releaseLocked() {
        _state.update { it.copy(modelReady = false) }
        while (_state.value.isBusy) {
            conversation?.cancelProcess()
            delay(100)
        }
        closeResources()
    }

    private fun closeResources() {
        try {
            conversation?.close()
        } catch (_: Exception) {
        }
        conversation = null

        try {
            engine?.close()
        } catch (_: Exception) {
        }
        engine = null
    }

    override fun onCleared() {
        scope.launch {
            mutex.withLock { releaseLocked() }
            executor.shutdown()
        }.invokeOnCompletion {
            scope.cancel()
        }
    }
}
