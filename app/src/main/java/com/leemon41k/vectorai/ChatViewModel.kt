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
import java.security.MessageDigest
import java.io.IOException
import java.util.UUID
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

data class ChatThread(
    val id: String,
    val title: String,
    val messages: List<ChatMessage>,
    val updatedAt: Long
)

data class ChatUiState(
    val modelReady: Boolean = false,
    val modelExists: Boolean = false,
    val isDownloading: Boolean = false,
    val isLoading: Boolean = false,
    val setupProgress: Float = 0f,
    val setupBytes: Long = 0L,
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
        private const val MODEL_FILE_NAME = "vectorai-qwen3-0.6b-int4.litertlm"
        private const val MODEL_ASSET_NAME = "vectorai-qwen3-0.6b-int4.litertlm"
        private const val EXPECTED_MODEL_SHA256 = "2df6821ec12702dafd33915e7a1a1adc7c4b053f3672fd9555dfaf3a114c4139"
        private const val REQUEST_LIMIT = 150
        private const val QUOTA_PREFS = "vectorai_request_quota"
    }

    private val executor = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(SupervisorJob() + executor.asCoroutineDispatcher())
    private val mutex = Mutex()
    private val modelFile = File(application.filesDir, MODEL_FILE_NAME)
    private val temporaryModelFile = File(application.filesDir, "$MODEL_FILE_NAME.tmp")
    private val quotaPrefs = application.getSharedPreferences(QUOTA_PREFS, android.content.Context.MODE_PRIVATE)
    private val chatPrefs = application.getSharedPreferences("vectorai_chats", android.content.Context.MODE_PRIVATE)
    private val initialThreads = loadThreadsFromDisk()

    private var engine: Engine? = null
    private var conversation: Conversation? = null

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private val _threads = MutableStateFlow(initialThreads)
    val threads: StateFlow<List<ChatThread>> = _threads.asStateFlow()

    private val _currentThreadId = MutableStateFlow(initialThreads.firstOrNull()?.id.orEmpty())
    val currentThreadId: StateFlow<String> = _currentThreadId.asStateFlow()

    private val _messages = MutableStateFlow(initialThreads.firstOrNull()?.messages ?: emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    init {
        if (initialThreads.isEmpty()) {
            val first = ChatThread(UUID.randomUUID().toString(), "Новый чат", emptyList(), System.currentTimeMillis())
            _threads.value = listOf(first)
            _currentThreadId.value = first.id
            persistThreads()
        }
    }

    fun newChat() {
        scope.launch {
            if (_state.value.isBusy || _messages.value.isNotEmpty()) return@launch
            val thread = ChatThread(UUID.randomUUID().toString(), "Новый чат", emptyList(), System.currentTimeMillis())
            _threads.update { listOf(thread) + it }
            _currentThreadId.value = thread.id
            _messages.value = emptyList()
            resetConversation()
            persistThreads()
        }
    }

    fun selectChat(id: String) {
        scope.launch {
            if (_state.value.isBusy) return@launch
            val thread = _threads.value.firstOrNull { it.id == id } ?: return@launch
            _currentThreadId.value = thread.id
            _messages.value = thread.messages
            resetConversation()
        }
    }

    fun deleteChat(id: String) {
        scope.launch {
            if (_state.value.isBusy) return@launch
            val remaining = _threads.value.filterNot { it.id == id }
            _threads.value = remaining
            if (_currentThreadId.value == id) {
                val next = remaining.firstOrNull()
                if (next == null) {
                    val replacement = ChatThread(UUID.randomUUID().toString(), "Новый чат", emptyList(), System.currentTimeMillis())
                    _threads.value = listOf(replacement)
                    _currentThreadId.value = replacement.id
                    _messages.value = emptyList()
                } else {
                    _currentThreadId.value = next.id
                    _messages.value = next.messages
                }
                resetConversation()
            }
            persistThreads()
        }
    }

    private fun resetConversation() {
        try { conversation?.close() } catch (_: Exception) { }
        conversation = try {
            engine?.createConversation(
                ConversationConfig(
                    systemInstruction = Contents.of(VECTORAI_SYSTEM_PROMPT),
                    samplerConfig = SamplerConfig(topK = 20, topP = 0.8, temperature = 0.7)
                )
            )
        } catch (_: Exception) { null }
    }

    private fun updateMessages(transform: (List<ChatMessage>) -> List<ChatMessage>) {
        _messages.update { transform(it) }
        val id = _currentThreadId.value
        val updatedMessages = _messages.value
        val firstPrompt = updatedMessages.firstOrNull { it.isUser }?.text
        val title = firstPrompt?.replace(Regex("\\s+"), " ")?.take(34) ?: "Новый чат"
        _threads.update { list ->
            list.map { thread ->
                if (thread.id == id) thread.copy(
                    title = title,
                    messages = updatedMessages,
                    updatedAt = System.currentTimeMillis()
                ) else thread
            }.sortedByDescending { it.updatedAt }
        }
        persistThreads()
    }

    private fun persistThreads() {
        val array = org.json.JSONArray()
        _threads.value.forEach { thread ->
            val item = org.json.JSONObject()
                .put("id", thread.id)
                .put("title", thread.title)
                .put("updatedAt", thread.updatedAt)
            val messagesArray = org.json.JSONArray()
            thread.messages.forEach { message ->
                messagesArray.put(org.json.JSONObject().put("isUser", message.isUser).put("text", message.text))
            }
            item.put("messages", messagesArray)
            array.put(item)
        }
        chatPrefs.edit().putString("threads", array.toString()).apply()
    }

    private fun loadThreadsFromDisk(): List<ChatThread> {
        return try {
            val raw = chatPrefs.getString("threads", null) ?: return emptyList()
            val array = org.json.JSONArray(raw)
            buildList {
                for (i in 0 until array.length()) {
                    val item = array.getJSONObject(i)
                    val messagesArray = item.optJSONArray("messages") ?: org.json.JSONArray()
                    val savedMessages = buildList {
                        for (j in 0 until messagesArray.length()) {
                            val message = messagesArray.getJSONObject(j)
                            add(ChatMessage(message.optBoolean("isUser"), message.optString("text")))
                        }
                    }
                    add(ChatThread(
                        id = item.optString("id"),
                        title = item.optString("title", "Новый чат"),
                        messages = savedMessages,
                        updatedAt = item.optLong("updatedAt", 0L)
                    ))
                }
            }.filter { it.id.isNotBlank() }.sortedByDescending { it.updatedAt }
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun prepare() {
        refreshQuota()
        scope.launch {
            mutex.withLock {
                if (_state.value.modelReady || _state.value.isLoading) return@withLock
                _state.update { it.copy(isLoading = true, isDownloading = false, error = null, setupProgress = 0f, setupBytes = 0L) }
                try {
                    val validExistingModel = modelFile.isFile &&
                        withContext(Dispatchers.IO) { sha256(modelFile) == EXPECTED_MODEL_SHA256 }
                    if (!validExistingModel) {
                        modelFile.delete()
                        temporaryModelFile.delete()
                        copyBundledModel()
                    } else {
                        _state.update { it.copy(modelExists = true, setupProgress = 1f) }
                    }
                    loadModelLocked()
                } catch (error: Exception) {
                    temporaryModelFile.delete()
                    _state.update {
                        it.copy(
                            modelReady = false,
                            modelExists = modelFile.isFile,
                            isLoading = false,
                            isDownloading = false,
                            error = error.localizedMessage ?: "Не удалось подготовить встроенную модель."
                        )
                    }
                }
            }
        }
    }

    private suspend fun copyBundledModel() = withContext(Dispatchers.IO) {
        val context = getApplication<Application>()
        val totalBytes = context.assets.openFd(MODEL_ASSET_NAME).use { it.length }
        val digest = MessageDigest.getInstance("SHA-256")
        var copiedBytes = 0L

        context.assets.open(MODEL_ASSET_NAME).buffered().use { input ->
            FileOutputStream(temporaryModelFile).buffered().use { output ->
                val buffer = ByteArray(256 * 1024)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    output.write(buffer, 0, count)
                    digest.update(buffer, 0, count)
                    copiedBytes += count
                    _state.update {
                        it.copy(
                            setupBytes = copiedBytes,
                            setupProgress = if (totalBytes > 0L) (copiedBytes.toFloat() / totalBytes).coerceIn(0f, 1f) else 0f
                        )
                    }
                }
                output.flush()
            }
        }

        val actualHash = digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
        if (actualHash != EXPECTED_MODEL_SHA256) {
            temporaryModelFile.delete()
            throw IOException("Встроенная модель повреждена: проверка SHA-256 не пройдена.")
        }
        if (modelFile.exists() && !modelFile.delete()) {
            throw IOException("Не удалось заменить старый файл модели.")
        }
        if (!temporaryModelFile.renameTo(modelFile)) {
            throw IOException("Не удалось сохранить встроенную модель.")
        }
        _state.update { it.copy(modelExists = true, setupProgress = 1f, setupBytes = copiedBytes) }
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
                updateMessages { it + ChatMessage(isUser = false, text = "Достигнут лимит запросов, попробуйте позже") }
                return@launch
            }

            updateMessages { it + ChatMessage(isUser = true, text = cleaned) + ChatMessage(isUser = false, text = "") }
            _state.update { it.copy(isBusy = true, error = null) }

            val answer = StringBuilder()
            try {
                val activeConversation = conversation ?: throw IllegalStateException("Локальная модель ещё не готова.")
                val knowledge = LocalKnowledgeBase.findRelevant(cleaned)
                val modelInput = if (knowledge.isBlank()) cleaned else
                    "Дополнительные сведения из локальной базы VectorAI. Используй их, если они относятся к вопросу; не выдумывай недостающие факты.\n$knowledge\n\nВопрос пользователя: $cleaned"
                activeConversation.sendMessageAsync(modelInput).collect { chunk ->
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

    fun refreshQuotaState() = refreshQuota()

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
        updateMessages { current ->
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
