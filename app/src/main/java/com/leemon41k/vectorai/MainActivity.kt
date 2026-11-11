package com.leemon41k.vectorai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import java.time.LocalTime

private val Black = Color(0xFF000000)
private val Panel = Color(0xFF101010)
private val Red = Color(0xFFFF343B)
private val SoftWhite = Color(0xFFEDEDED)
private val Muted = Color(0xFF858585)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = android.graphics.Color.BLACK
        window.navigationBarColor = android.graphics.Color.BLACK
        window.decorView.systemUiVisibility = 0
        setContent {
            MaterialTheme {
                VectorAIApp()
            }
        }
    }
}

@Composable
private fun VectorAIApp() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val preferences = remember {
        context.getSharedPreferences("vectorai_settings", android.content.Context.MODE_PRIVATE)
    }
    var userName by rememberSaveable {
        mutableStateOf(preferences.getString("user_name", "").orEmpty())
    }
    var currentPage by rememberSaveable { mutableStateOf("chat") }
    val chatViewModel: ChatViewModel = viewModel()
    val chatState by chatViewModel.state.collectAsState()
    val messages by chatViewModel.messages.collectAsState()
    LaunchedEffect(Unit) {
        chatViewModel.prepare()
    }

    BackHandler(enabled = currentPage != "chat") {
        currentPage = "chat"
    }

    Box(modifier = Modifier.fillMaxSize().background(Black)) {
        AnimatedGeometryBackground()

        if (userName.isBlank()) {
            NameSetupScreen(
                onContinue = { enteredName ->
                    val cleaned = enteredName.trim()
                    if (cleaned.isNotEmpty()) {
                        preferences.edit().putString("user_name", cleaned).apply()
                        userName = cleaned
                    }
                }
            )
        } else {
            MainNavigation(
                userName = userName,
                currentPage = currentPage,
                onPageChange = { currentPage = it },
                messages = messages,
                chatState = chatState,
                onSendMessage = chatViewModel::send,
                onDownloadModel = chatViewModel::startDownload
            )
        }

    }
}

@Composable
private fun NameSetupScreen(onContinue: (String) -> Unit) {
    var name by rememberSaveable { mutableStateOf("") }
    Column(
        modifier = Modifier.fillMaxSize().padding(horizontal = 28.dp),
        verticalArrangement = Arrangement.Center
    ) {
        Text("VECTORAI", color = SoftWhite, fontSize = 13.sp, letterSpacing = 5.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(34.dp))
        Text("Как к вам обращаться?", color = SoftWhite, fontSize = 29.sp, lineHeight = 35.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(12.dp))
        Text("Это имя будет использоваться в приветствиях.", color = Muted, fontSize = 14.sp)
        Spacer(Modifier.height(28.dp))
        OutlinedTextField(
            value = name,
            onValueChange = { name = it },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            placeholder = { Text("Ваше имя", color = Muted) },
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
            keyboardActions = KeyboardActions(onDone = { onContinue(name) }),
            colors = vectorFieldColors()
        )
        Spacer(Modifier.height(18.dp))
        Button(
            onClick = { onContinue(name) },
            enabled = name.trim().isNotEmpty(),
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(14.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Red, disabledContainerColor = Color(0xFF552326))
        ) {
            Text("Продолжить", fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
private fun MainNavigation(
    userName: String,
    currentPage: String,
    onPageChange: (String) -> Unit,
    messages: List<ChatMessage>,
    chatState: ChatUiState,
    onSendMessage: (String) -> Unit,
    onDownloadModel: () -> Unit
) {
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    ModalNavigationDrawer(
        drawerState = drawerState,
        scrimColor = Color.Black.copy(alpha = 0.72f),
        drawerContent = {
            ModalDrawerSheet(
                modifier = Modifier.width(300.dp),
                drawerContainerColor = Panel,
                windowInsets = WindowInsets.navigationBars
            ) {
                Column(modifier = Modifier.fillMaxSize().statusBarsPadding().padding(20.dp)) {
                    Text("НАСТРОЙКИ", color = Muted, fontSize = 11.sp, letterSpacing = 2.sp, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(12.dp))
                    DrawerEntry("Настройки", selected = currentPage == "settings") {
                        onPageChange("settings")
                        scope.launch { drawerState.close() }
                    }
                    Spacer(Modifier.height(28.dp))
                    Text("ИСТОРИЯ ЧАТОВ", color = Muted, fontSize = 11.sp, letterSpacing = 2.sp, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(12.dp))
                    DrawerEntry("Новый чат", selected = currentPage == "chat") {
                        onPageChange("chat")
                        scope.launch { drawerState.close() }
                    }
                    Text(
                        if (messages.isEmpty()) "История пока пуста" else "Текущий диалог",
                        color = Muted,
                        fontSize = 13.sp,
                        modifier = Modifier.padding(start = 14.dp, top = 8.dp)
                    )
                    Spacer(Modifier.weight(1f))
                    Divider(color = Color.White.copy(alpha = 0.12f))
                    Spacer(Modifier.height(14.dp))
                    Text(
                        "Версия приложения 0.2.0",
                        color = Color(0xFF777777),
                        fontSize = 12.sp,
                        modifier = Modifier.padding(vertical = 8.dp, horizontal = 2.dp)
                    )
                }
            }
        }
    ) {
        when (currentPage) {
            "settings" -> SettingsScreen(userName = userName)
            else -> ChatScreen(
                userName = userName,
                onMenu = { scope.launch { drawerState.open() } },
                messages = messages,
                chatState = chatState,
                onSendMessage = onSendMessage,
                onDownloadModel = onDownloadModel
            )
        }
    }
}

@Composable
private fun DrawerEntry(title: String, selected: Boolean, onClick: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp))
            .background(if (selected) Color.White.copy(alpha = 0.07f) else Color.Transparent)
            .clickable(onClick = onClick).padding(horizontal = 14.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(Modifier.size(6.dp).clip(CircleShape).background(if (selected) Red else Muted))
        Spacer(Modifier.width(12.dp))
        Text(title, color = if (selected) SoftWhite else Color(0xFFB0B0B0), fontSize = 15.sp)
    }
}

@Composable
private fun ChatScreen(
    userName: String,
    onMenu: () -> Unit,
    messages: List<ChatMessage>,
    chatState: ChatUiState,
    onSendMessage: (String) -> Unit,
    onDownloadModel: () -> Unit
) {
    var draft by rememberSaveable { mutableStateOf("") }
    val greeting = greetingForCurrentTime()

    Column(
        modifier = Modifier.fillMaxSize()
            .statusBarsPadding()
            .imePadding()
            .padding(horizontal = 18.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 16.dp, bottom = 20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Canvas(
                modifier = Modifier.size(44.dp).clickable(onClick = onMenu)
            ) {
                val lineColor = SoftWhite
                drawLine(lineColor, androidx.compose.ui.geometry.Offset(3f, size.height * 0.34f), androidx.compose.ui.geometry.Offset(size.width - 3f, size.height * 0.34f), strokeWidth = 2.2f)
                drawLine(lineColor, androidx.compose.ui.geometry.Offset(3f, size.height * 0.68f), androidx.compose.ui.geometry.Offset(size.width - 3f, size.height * 0.68f), strokeWidth = 2.2f)
            }
        }

        if (messages.isEmpty()) {
            Column(modifier = Modifier.weight(1f).fillMaxWidth(), verticalArrangement = Arrangement.Center) {
                Text(greeting, color = SoftWhite, fontSize = 27.sp, fontWeight = FontWeight.Medium, lineHeight = 34.sp)
                Text(userName, color = Red, fontSize = 29.sp, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(14.dp))
                Text("Чем займёмся сегодня?", color = Muted, fontSize = 15.sp)

                if (!chatState.modelReady) {
                    Spacer(Modifier.height(28.dp))
                    when {
                        chatState.isDownloading -> {
                            Text("Загружаю локальную модель…", color = SoftWhite, fontSize = 14.sp)
                            Spacer(Modifier.height(12.dp))
                            LinearProgressIndicator(
                                progress = chatState.downloadProgress,
                                modifier = Modifier.fillMaxWidth(),
                                color = Red,
                                trackColor = Color.White.copy(alpha = 0.12f)
                            )
                            Spacer(Modifier.height(8.dp))
                            val downloadedMb = chatState.downloadedBytes / 1_000_000
                            Text("$downloadedMb / ${chatState.totalBytes / 1_000_000} МБ · ${(chatState.downloadProgress * 100).toInt()}%", color = Muted, fontSize = 12.sp)
                            val speedMb = chatState.downloadSpeed / 1_000_000.0
                            val eta = chatState.remainingSeconds
                            Text(
                                (if (speedMb > 0) String.format(java.util.Locale.getDefault(), "%.1f МБ/с", speedMb) else "Скорость рассчитывается…") +
                                    if (eta >= 0L) " · осталось ${eta / 60} мин ${eta % 60} сек" else "",
                                color = Muted,
                                fontSize = 12.sp
                            )
                        }
                        chatState.isLoading -> {
                            Text("Запускаю VectorAI…", color = Muted, fontSize = 14.sp)
                            Spacer(Modifier.height(10.dp))
                            LinearProgressIndicator(
                                modifier = Modifier.fillMaxWidth(),
                                color = Red,
                                trackColor = Color.White.copy(alpha = 0.12f)
                            )
                        }
                        else -> {
                            Text(
                                "Для первого запуска нужно скачать локальную модель VectorAI (около 1,6 ГБ). Загрузка происходит прямо в приложении. После установки помощник сможет работать офлайн.",
                                color = Muted,
                                fontSize = 13.sp,
                                lineHeight = 19.sp
                            )
                            chatState.error?.let {
                                Spacer(Modifier.height(8.dp))
                                Text(it, color = Color(0xFFFF7777), fontSize = 12.sp, lineHeight = 17.sp)
                            }
                            Spacer(Modifier.height(14.dp))
                            Button(
                                onClick = onDownloadModel,
                                modifier = Modifier.fillMaxWidth().height(50.dp),
                                shape = RoundedCornerShape(14.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = Red)
                            ) {
                                Text(
                                    if (chatState.downloadedBytes > 0L) "Продолжить загрузку" else "Скачать модель",
                                    color = Color.White,
                                    fontWeight = FontWeight.SemiBold
                                )
                            }
                        }
                    }
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(14.dp),
                reverseLayout = false
            ) {
                item {
                    Text("$greeting, $userName", color = Muted, fontSize = 13.sp, modifier = Modifier.padding(vertical = 8.dp))
                }
                itemsIndexed(messages) { _, message ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = if (message.isUser) Arrangement.End else Arrangement.Start
                    ) {
                        Box(
                            modifier = Modifier.fillMaxWidth(0.88f)
                                .clip(RoundedCornerShape(16.dp))
                                .background(if (message.isUser) Color(0xFF251113) else Color(0xFF171717))
                                .padding(14.dp)
                        ) {
                            Text(message.text, color = SoftWhite, fontSize = 14.sp, lineHeight = 20.sp)
                        }
                    }
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 12.dp, bottom = 14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value = draft,
                onValueChange = { draft = it },
                enabled = chatState.modelReady && !chatState.isBusy,
                modifier = Modifier.weight(1f),
                placeholder = { Text("Сообщение VectorAI...", color = Muted, fontSize = 14.sp) },
                shape = RoundedCornerShape(18.dp),
                maxLines = 4,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = {
                    if (draft.isNotBlank() && chatState.modelReady && !chatState.isBusy) {
                        onSendMessage(draft.trim())
                        draft = ""
                    }
                }),
                colors = vectorFieldColors()
            )
            Spacer(Modifier.width(8.dp))
            Box(
                modifier = Modifier.size(50.dp).clip(CircleShape).background(Red)
                    .clickable {
                        if (draft.isNotBlank() && chatState.modelReady && !chatState.isBusy) {
                            onSendMessage(draft.trim())
                            draft = ""
                        }
                    },
                contentAlignment = Alignment.Center
            ) {
                Text("↑", color = Color.White, fontSize = 27.sp, fontWeight = FontWeight.Medium)
            }
        }
    }
}

@Composable
private fun SettingsScreen(userName: String) {
    Column(modifier = Modifier.fillMaxSize().statusBarsPadding().padding(top = 22.dp, start = 22.dp, end = 22.dp, bottom = 22.dp)) {
        Text("Настройки", color = SoftWhite, fontSize = 28.sp, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(24.dp))
        Text("ПРОФИЛЬ", color = Muted, fontSize = 11.sp, letterSpacing = 2.sp)
        Spacer(Modifier.height(10.dp))
        Text("Имя", color = Muted, fontSize = 13.sp)
        Text(userName, color = SoftWhite, fontSize = 17.sp)
        Spacer(Modifier.height(24.dp))
        Divider(color = Color.White.copy(alpha = 0.12f))
        Spacer(Modifier.height(18.dp))
        Text("Внешний вид", color = SoftWhite, fontSize = 16.sp)
        Text("Чёрная тема с красными акцентами", color = Muted, fontSize = 13.sp, modifier = Modifier.padding(top = 6.dp))
        Spacer(Modifier.height(18.dp))
        Text("Анимация фона", color = SoftWhite, fontSize = 16.sp)
        Text("Геометрические элементы активны", color = Muted, fontSize = 13.sp, modifier = Modifier.padding(top = 6.dp))
    }
}

@Composable
private fun AnimatedGeometryBackground() {
    val transition = rememberInfiniteTransition(label = "geometry")
    val phase by transition.animateFloat(
        initialValue = 0f,
        targetValue = (Math.PI * 2.0).toFloat(),
        animationSpec = infiniteRepeatable(animation = tween(18000, easing = LinearEasing), repeatMode = RepeatMode.Restart),
        label = "phase"
    )

    Canvas(modifier = Modifier.fillMaxSize().alpha(0.92f)) {
        val w = size.width
        val h = size.height
        for (i in 0 until 54) {
            val baseX = ((i * 79) % 101) / 100f * w
            val baseY = ((i * 137) % 103) / 103f * h
            val drift = kotlin.math.sin(phase + i * 0.53f)
            val x = baseX + drift * 23f
            val y = baseY + kotlin.math.cos(phase * 0.7f + i) * 17f
            val length = 22f + ((i * 17) % 48)
            val angle = kotlin.math.sin(phase * 0.6f + i * 0.41f) * 0.8f
            val endX = x + kotlin.math.sin(angle) * length
            val endY = y + kotlin.math.cos(angle) * length
            val color = if (i % 3 == 0 || i % 3 == 1) Red.copy(alpha = 0.17f) else Color.White.copy(alpha = 0.13f)
            drawLine(color = color, start = androidx.compose.ui.geometry.Offset(x, y), end = androidx.compose.ui.geometry.Offset(endX, endY), strokeWidth = if (i % 5 == 0) 2.1f else 1.25f)
            if (i % 4 == 0) {
                drawCircle(color = color.copy(alpha = 0.8f), radius = 2.2f + (i % 3), center = androidx.compose.ui.geometry.Offset(endX, endY))
            }
            if (i % 9 == 0) {
                val radius = 12f + (i % 4) * 4f
                drawCircle(color = Red.copy(alpha = 0.075f), radius = radius, center = androidx.compose.ui.geometry.Offset(x + 15f, y - 10f), style = Stroke(width = 1.2f))
            }
        }

        // A few slowly morphing angular outlines give the background a geometric rhythm.
        for (i in 0 until 5) {
            val cx = w * (0.15f + i * 0.18f) + kotlin.math.sin(phase + i) * 12f
            val cy = h * (0.22f + (i % 3) * 0.25f) + kotlin.math.cos(phase + i * 0.7f) * 15f
            val r = 18f + i * 5f
            val path = Path().apply {
                moveTo(cx, cy - r)
                lineTo(cx + r, cy)
                lineTo(cx, cy + r)
                lineTo(cx - r, cy)
                close()
            }
            drawPath(path, color = if (i % 2 == 0) Red.copy(alpha = 0.10f) else Color.White.copy(alpha = 0.07f), style = Stroke(width = 1.1f))
        }
    }
}

@Composable
private fun vectorFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedTextColor = SoftWhite,
    unfocusedTextColor = SoftWhite,
    focusedBorderColor = Red,
    unfocusedBorderColor = Color.White.copy(alpha = 0.22f),
    focusedLabelColor = Red,
    unfocusedLabelColor = Muted,
    cursorColor = Red,
    focusedPlaceholderColor = Muted,
    unfocusedPlaceholderColor = Muted
)

private fun greetingForCurrentTime(): String {
    val hour = LocalTime.now().hour
    return when (hour) {
        in 5..11 -> "Доброе утро"
        in 12..17 -> "Добрый день"
        in 18..22 -> "Добрый вечер"
        else -> "Доброй ночи"
    }
}
