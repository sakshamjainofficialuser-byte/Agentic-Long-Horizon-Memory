/**
 * Agentic Long-Horizon Memory Dashboard Client Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  const statusBadge = document.getElementById("system-status");
  const statusText = document.getElementById("status-text");
  const modelName = document.getElementById("model-name");
  const activeSessionInput = document.getElementById("active-session-id");
  const btnSwitchSession = document.getElementById("btn-switch-session");
  const btnNewSession = document.getElementById("btn-new-session");
  const btnClearChat = document.getElementById("btn-clear-chat");
  const sessionsList = document.getElementById("sessions-list");
  const chatSessionTag = document.getElementById("chat-session-tag");
  const messagesContainer = document.getElementById("messages-container");
  const chatInput = document.getElementById("chat-input");
  const btnSend = document.getElementById("btn-send");
  const statTurns = document.getElementById("stat-turns");

  // Inspector elements
  const layerSystemContent = document.getElementById("layer-system-content");
  const layerMemoriesContent = document.getElementById("layer-memories-content");
  const memoriesCountBadge = document.getElementById("memories-count-badge");
  const layerRecentContent = document.getElementById("layer-recent-content");
  const recentCountBadge = document.getElementById("recent-count-badge");
  const layerUserContent = document.getElementById("layer-user-content");
  const rawJsonViewer = document.getElementById("raw-json-viewer");

  let currentSessionId = activeSessionInput.value.trim() || "session_default";
  let turnCount = 0;

  // 1. Check System Health
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();

      statusBadge.classList.add("online");
      statusText.textContent = data.has_api_key ? "Agent Online" : "Online (No API Key)";
      modelName.textContent = `Model: ${data.model}`;
    } catch (err) {
      statusBadge.classList.remove("online");
      statusText.textContent = "Server Offline";
    }
  }

  // 2. Fetch Sessions List
  async function loadSessions() {
    try {
      const res = await fetch("/api/sessions");
      if (!res.ok) return;
      const sessions = await res.json();
      if (!sessions || sessions.length === 0) return;

      sessionsList.innerHTML = "";
      sessions.forEach(sess => {
        const item = document.createElement("div");
        item.className = `session-item ${sess.session_id === currentSessionId ? 'active' : ''}`;
        item.innerHTML = `
          <span class="session-icon">💬</span>
          <div class="session-info">
            <span class="session-title">${escapeHtml(sess.session_id)}</span>
            <span class="session-meta">${sess.turns} turns</span>
          </div>
        `;
        item.addEventListener("click", () => switchSession(sess.session_id));
        sessionsList.appendChild(item);
      });
    } catch (err) {
      console.warn("Could not load sessions", err);
    }
  }

  function switchSession(newSessionId) {
    currentSessionId = newSessionId;
    activeSessionInput.value = currentSessionId;
    chatSessionTag.textContent = `Session: ${currentSessionId}`;
    messagesContainer.innerHTML = `
      <div class="message-bubble system-welcome">
        <div class="avatar">🤖</div>
        <div class="bubble-content">
          <strong>Switched to Session: ${escapeHtml(currentSessionId)}</strong>
          <p>You are now interacting in this isolated conversation context.</p>
        </div>
      </div>
    `;
    loadSessions();
  }

  // 3. Update Context Inspector
  function updateContextInspector(contextArray, userMessage) {
    if (!contextArray || !Array.isArray(contextArray)) return;

    rawJsonViewer.textContent = JSON.stringify(contextArray, null, 2);

    // Layer 1: System instruction
    const sysMsg = contextArray.find(m => m.role === "system" && !m.content.includes("Relevant background memories:"));
    if (sysMsg) {
      layerSystemContent.textContent = sysMsg.content;
    }

    // Layer 2: Memories
    const memMsg = contextArray.find(m => m.role === "system" && m.content.includes("Relevant background memories:"));
    if (memMsg) {
      layerMemoriesContent.textContent = memMsg.content;
      memoriesCountBadge.textContent = "Retrieved";
    } else {
      layerMemoriesContent.innerHTML = '<span class="empty-placeholder">No semantic memories retrieved for this turn.</span>';
      memoriesCountBadge.textContent = "0 Items";
    }

    // Layer 3: Recent turns
    const recentMsgs = contextArray.filter(m => m.role !== "system").slice(0, -1);
    if (recentMsgs.length > 0) {
      recentCountBadge.textContent = `${recentMsgs.length} Items`;
      layerRecentContent.textContent = recentMsgs.map(m => `[${m.role}]: ${m.content}`).join("\n\n");
    } else {
      recentCountBadge.textContent = "0 Items";
      layerRecentContent.innerHTML = '<span class="empty-placeholder">No prior conversation turns in context.</span>';
    }

    // Layer 4: Current user prompt
    layerUserContent.textContent = userMessage || (contextArray[contextArray.length - 1]?.content ?? "");
  }

  // 4. Send Message
  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = "";
    btnSend.disabled = true;

    // Append user bubble
    appendMessage("user", text);

    // Append typing indicator
    const typingId = appendTypingIndicator();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: currentSessionId,
        }),
      });

      removeTypingIndicator(typingId);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: "Network error" }));
        appendMessage("assistant", `⚠️ Error: ${errData.detail || res.statusText}`);
        return;
      }

      const data = await res.json();
      appendMessage("assistant", data.response);

      // Increment stats
      turnCount += 1;
      statTurns.textContent = turnCount;

      // Update Inspector
      updateContextInspector(data.context, text);

      // Refresh sessions
      loadSessions();
    } catch (err) {
      removeTypingIndicator(typingId);
      appendMessage("assistant", `⚠️ System Error: ${err.message}`);
    } finally {
      btnSend.disabled = false;
      chatInput.focus();
    }
  }

  function appendMessage(role, text) {
    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${role}`;
    const avatarIcon = role === "user" ? "👤" : "🤖";
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    bubble.innerHTML = `
      <div class="avatar">${avatarIcon}</div>
      <div class="bubble-content">
        <p>${escapeHtml(text)}</p>
        <span class="bubble-time">${timeStr}</span>
      </div>
    `;
    messagesContainer.appendChild(bubble);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function appendTypingIndicator() {
    const id = "typing-" + Date.now();
    const bubble = document.createElement("div");
    bubble.className = "message-bubble assistant";
    bubble.id = id;
    bubble.innerHTML = `
      <div class="avatar">🤖</div>
      <div class="bubble-content">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    messagesContainer.appendChild(bubble);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return id;
  }

  function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // Event Listeners
  btnSend.addEventListener("click", sendMessage);

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  btnSwitchSession.addEventListener("click", () => {
    const val = activeSessionInput.value.trim();
    if (val) switchSession(val);
  });

  btnNewSession.addEventListener("click", () => {
    const newId = "session_" + Math.random().toString(36).substring(2, 9);
    switchSession(newId);
  });

  btnClearChat.addEventListener("click", () => {
    messagesContainer.innerHTML = `
      <div class="message-bubble system-welcome">
        <div class="avatar">🤖</div>
        <div class="bubble-content">
          <strong>Screen Cleared</strong>
          <p>Chat display reset. History remains saved in session memory.</p>
        </div>
      </div>
    `;
  });

  // Initial load
  checkHealth();
  loadSessions();
});
