package com.leemon41k.vectorai

import android.app.Application
import android.net.Uri
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
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.collect
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
        private const val MODEL_FILE_NAME = "vectorai-gemma3-1b-it-int4.litertlm"
        private const val MIN_MODEL_SIZE = 570_000_000L
        private const val MAX_MODEL_SIZE = 610_000_000L
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

    fun retryLoadingModel() {
        scope.launch {
            mutex.withLock {
                if (modelFile.isFile && modelFile.length() in MIN_MODEL_SIZE..MAX_MODEL_SIZE) {
                    loadModelLocked()
                } else {
                    _state.update { it.copy(error = "Сначала выбери файл Gemma 3 (.litertlm).") }
                }
            }
        }
    }

    fun importModel(uri: Uri) {
        scope.launch {
            mutex.withLock {
                if (_state.value.isDownloading || _state.value.isLoading) return@withLock
                releaseLocked()
                val partialFile = File(getApplication<Application>().cacheDir, "$MODEL_FILE_NAME.part")
                _state.value = ChatUiState(isDownloading = true, error = null)
                try {
                    withContext(Dispatchers.IO) {
                        val resolver = getApplication<Application>().contentResolver
                        val input = resolver.openInputStream(uri)
                            ?: throw IOException("Не удалось открыть выбранный файл.")
                        input.use { source ->
                            FileOutputStream(partialFile).use { output ->
                                val buffer = ByteArray(1024 * 1024)
                                var copied = 0L
                                while (true) {
                                    val count = source.read(buffer)
                                    if (count < 0) break
                                    output.write(buffer, 0, count)
                                    copied += count
                                    _state.update {
                                        it.copy(
                                            downloadedBytes = copied,
                                            totalBytes = 584_000_000L,
                                            downloadProgress = (copied.toFloat() / 584_000_000f).coerceIn(0f, 1f)
                                        )
                                    }
                                }
                                output.fd.sync()
                            }
                        }
                    }
                    if (partialFile.length() !in MIN_MODEL_SIZE..MAX_MODEL_SIZE) {
                        throw IOException("Неверный файл. Нужен gemma3-1b-it-int4.litertlm размером около 584 МБ.")
                    }
                    if (modelFile.exists() && !modelFile.delete()) {
                        throw IOException("Не удалось заменить предыдущий файл модели.")
                    }
                    if (!partialFile.renameTo(modelFile)) {
                        throw IOException("Не удалось сохранить файл Gemma.")
                    }
                    File(getApplication<Application>().filesDir, "vectorai-qwen3-0.6b.litertlm").delete()
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
                            error = error.localizedMessage ?: "Не удалось импортировать модель."
                        )
                    }
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
                activeConversation.sendMessageAsync(
                    cleaned,
                    extraContext = mapOf("enable_thinking" to false)
                ).collect { chunk ->
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
                        topK = 40,
                        topP = 0.9,
                        temperature = 0.3
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
