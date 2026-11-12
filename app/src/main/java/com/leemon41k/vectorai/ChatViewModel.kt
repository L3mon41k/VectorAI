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
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
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
    val downloadSpeed: Long = 0L,
    val remainingSeconds: Long = -1L,
    val error: String? = null,
    val requestsUsed: Int = 0,
    val requestsRemaining: Int = 150,
    val limitReached: Boolean = false
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    companion object {
        private const val MODEL_FILE_NAME = "vectorai-local-model.litertlm"
        private const val PART_FILE_NAME = "$MODEL_FILE_NAME.part"
        private const val EXPECTED_MODEL_SIZE = 1_597_931_520L
        private const val EXPECTED_MODEL_SHA256 = "c4a9b7289365ab80fe599b844b77ad253eb3c53fb0e5d963d699146a6f01255e"
        private const val MODEL_URL = "https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct/resolve/main/Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm"
        private const val REQUEST_LIMIT = 150
        private const val QUOTA_PREFS = "vectorai_request_quota"
    }

    private val executor = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(SupervisorJob() + executor.asCoroutineDispatcher())
    private val mutex = Mutex()
    private val modelFile = File(application.filesDir, MODEL_FILE_NAME)
    private val partialFile = File(application.filesDir, PART_FILE_NAME)
    private val quotaPrefs = application.getSharedPreferences(QUOTA_PREFS, android.content.Context.MODE_PRIVATE)

    private var engine: Engine? = null
    private var conversation: Conversation? = null

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    fun prepare() {
        refreshQuota()
        scope.launch {
            mutex.withLock {
                if (_state.value.modelReady || _state.value.isLoading || _state.value.isDownloading) {
                    return@withLock
                }
                if (modelFile.isFile && modelFile.length() == EXPECTED_MODEL_SIZE && withContext(Dispatchers.IO) { sha256(modelFile) == EXPECTED_MODEL_SHA256 }) {
                    _state.update { it.copy(modelExists = true, error = null) }
                    loadModelLocked()
                } else {
                    modelFile.delete()
                    _state.value = ChatUiState()
                }
            }
        }
    }

    fun startDownload() {
        scope.launch {
            mutex.withLock {
                if (_state.value.isDownloading || _state.value.isLoading || _state.value.modelReady) return@withLock
                try {
                    _state.update { it.copy(isDownloading = true, isLoading = false, modelExists = false, downloadedBytes = partialFile.length().coerceAtMost(EXPECTED_MODEL_SIZE), totalBytes = EXPECTED_MODEL_SIZE, downloadProgress = (partialFile.length().toFloat() / EXPECTED_MODEL_SIZE).coerceIn(0f, 1f), downloadSpeed = 0L, remainingSeconds = -1L, error = null) }
                    downloadModel()
                    val hash = withContext(Dispatchers.IO) { sha256(partialFile) }
                    if (hash != EXPECTED_MODEL_SHA256) { partialFile.delete(); throw IOException("Проверка SHA-256 не пройдена. Загрузи файл ещё раз.") }
                    if (modelFile.exists() && !modelFile.delete()) throw IOException("Не удалось заменить файл модели.")
                    if (!partialFile.renameTo(modelFile)) throw IOException("Не удалось сохранить файл модели.")
                    _state.update { it.copy(isDownloading = false, modelExists = true, isLoading = true) }
                    loadModelLocked()
                    if (_state.value.modelReady) File(getApplication<Application>().filesDir, "vectorai-gemma3-1b-it-int4.litertlm").delete()
                } catch (error: Exception) {
                    _state.update { it.copy(isDownloading = false, isLoading = false, modelReady = false, modelExists = modelFile.isFile, error = error.localizedMessage ?: "Не удалось загрузить модель. Незавершённая загрузка сохранена.") }
                }
            }
        }
    }

    private suspend fun downloadModel() = withContext(Dispatchers.IO) {
        downloadModelAttempt(retryAfter416 = true)
    }

    private suspend fun downloadModelAttempt(retryAfter416: Boolean) {
        var offset = partialFile.length()
        if (offset > EXPECTED_MODEL_SIZE) {
            partialFile.delete()
            offset = 0L
        }
        if (offset == EXPECTED_MODEL_SIZE) return

        val connection = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = true
            connectTimeout = 30_000
            readTimeout = 60_000
            setRequestProperty("User-Agent", "VectorAI-Android")
            setRequestProperty("Accept-Encoding", "identity")
            if (offset > 0L) setRequestProperty("Range", "bytes=$offset-")
        }

        try {
            connection.connect()
            val status = connection.responseCode

            if (status == 416 && retryAfter416) {
                connection.disconnect()
                partialFile.delete()
                return downloadModelAttempt(retryAfter416 = false)
            }

            val append = offset > 0L && status == HttpURLConnection.HTTP_PARTIAL
            if (status != HttpURLConnection.HTTP_OK && !append) {
                throw IOException("Ошибка сервера: HTTP $status.")
            }

            if (append) {
                val contentRange = connection.getHeaderField("Content-Range")
                    ?: throw IOException("Сервер не подтвердил диапазон докачки.")
                val match = Regex("bytes (\\d+)-(\\d+)/(\\d+)").matchEntire(contentRange.trim())
                    ?: throw IOException("Некорректный Content-Range от сервера.")
                val rangeStart = match.groupValues[1].toLong()
                val rangeEnd = match.groupValues[2].toLong()
                val remoteTotal = match.groupValues[3].toLong()
                if (rangeStart != offset || rangeEnd < rangeStart || remoteTotal != EXPECTED_MODEL_SIZE) {
                    connection.disconnect()
                    partialFile.delete()
                    if (retryAfter416) return downloadModelAttempt(retryAfter416 = false)
                    throw IOException("Диапазон загрузки не совпадает с локальным файлом.")
                }
            } else if (offset > 0L && status == HttpURLConnection.HTTP_OK) {
                offset = 0L
            }

            val responseLength = connection.contentLengthLong
            if (responseLength > 0L && offset + responseLength > EXPECTED_MODEL_SIZE) {
                throw IOException("Сервер передал файл неожиданного размера.")
            }

            FileOutputStream(partialFile, append).use { output ->
                connection.inputStream.use { input ->
                    val buffer = ByteArray(262144)
                    var countTotal = offset
                    var lastTime = System.nanoTime()
                    var lastCount = countTotal
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        output.write(buffer, 0, n)
                        countTotal += n
                        if (countTotal > EXPECTED_MODEL_SIZE) {
                            throw IOException("Получен файл неожиданного размера.")
                        }
                        val now = System.nanoTime()
                        if (now - lastTime >= 500_000_000L) {
                            val seconds = (now - lastTime) / 1_000_000_000.0
                            val speed = ((countTotal - lastCount) / seconds).toLong().coerceAtLeast(0L)
                            val eta = if (speed > 0L) (EXPECTED_MODEL_SIZE - countTotal) / speed else -1L
                            _state.update {
                                it.copy(
                                    downloadedBytes = countTotal,
                                    totalBytes = EXPECTED_MODEL_SIZE,
                                    downloadProgress = (countTotal.toFloat() / EXPECTED_MODEL_SIZE).coerceIn(0f, 1f),
                                    downloadSpeed = speed,
                                    remainingSeconds = eta
                                )
                            }
                            lastTime = now
                            lastCount = countTotal
                        }
                    }
                    output.fd.sync()
                    if (countTotal != EXPECTED_MODEL_SIZE) {
                        throw IOException("Загружено ${countTotal / 1_000_000} из ${EXPECTED_MODEL_SIZE / 1_000_000} МБ. Нажми «Продолжить загрузку».")
                    }
                }
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) { val n = input.read(buffer); if (n < 0) break; digest.update(buffer, 0, n) }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    fun send(text: String) {
        val cleaned = text.trim()
        if (cleaned.isEmpty()) return

        scope.launch {
            if (!_state.value.modelReady || _state.value.isBusy) return@launch
            if (!consumeRequest()) {
                _messages.update { it + ChatMessage(isUser = false, text = "Достигнут лимит запросов, попробуйте позже") }
                return@launch
            }

            _messages.update { it + ChatMessage(isUser = true, text = cleaned) + ChatMessage(isUser = false, text = "") }
            _state.update { it.copy(isBusy = true, error = null) }

            val answer = StringBuilder()
            try {
                val activeConversation = conversation ?: throw IllegalStateException("Локальная модель ещё не готова.")
                val knowledge = LocalKnowledgeBase.findRelevant(cleaned)
                val modelInput = if (knowledge.isBlank()) cleaned else
                    "Дополнительные сведения из локальной базы VectorAI. Используй их, если они относятся к вопросу; не выдумывай недостающие факты.\\n$knowledge\\n\\nВопрос пользователя: $cleaned"
                activeConversation.sendMessageAsync(
                    modelInput,
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

    private fun currentQuotaPeriodKey(): String {
        val calendar = java.util.Calendar.getInstance()
        val year = calendar.get(java.util.Calendar.YEAR)
        val day = calendar.get(java.util.Calendar.DAY_OF_YEAR)
        val periodStartHour = (calendar.get(java.util.Calendar.HOUR_OF_DAY) / 6) * 6
        return "$year-$day-$periodStartHour"
    }

    private fun refreshQuota() {
        val key = currentQuotaPeriodKey()
        val storedKey = quotaPrefs.getString("period_key", null)
        var used = quotaPrefs.getInt("used", 0)
        if (storedKey != key) {
            used = 0
            quotaPrefs.edit().putString("period_key", key).putInt("used", 0).apply()
        }
        _state.update {
            it.copy(
                requestsUsed = used.coerceIn(0, REQUEST_LIMIT),
                requestsRemaining = (REQUEST_LIMIT - used).coerceIn(0, REQUEST_LIMIT),
                limitReached = used >= REQUEST_LIMIT
            )
        }
    }

    private fun consumeRequest(): Boolean {
        refreshQuota()
        val used = quotaPrefs.getInt("used", 0)
        if (used >= REQUEST_LIMIT) return false
        val next = used + 1
        quotaPrefs.edit().putString("period_key", currentQuotaPeriodKey()).putInt("used", next).commit()
        _state.update {
            it.copy(
                requestsUsed = next,
                requestsRemaining = REQUEST_LIMIT - next,
                limitReached = next >= REQUEST_LIMIT
            )
        }
        return true
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
                        topK = 20,
                        topP = 0.8,
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
