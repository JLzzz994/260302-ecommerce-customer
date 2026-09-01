<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";

type Role = "user" | "bot" | "agent" | "system";

type ChatMessage = {
  role: Role;
  text: string;
};

type QueueItem = {
  sender_id: string;
  position: number;
};

const mode = ref<"user" | "agent">("user");
const messages = ref<ChatMessage[]>([]);
const input = ref("");
const connected = ref(false);
const transferMode = ref<"machine" | "queue" | "human">("machine");
const queuePosition = ref<number | null>(null);
const currentAgent = ref<string | null>(null);

const senderId = ref(
  localStorage.getItem("wd_sender_id") ||
    `buyer-${Math.random().toString(36).slice(2, 8)}`,
);
localStorage.setItem("wd_sender_id", senderId.value);

const agentId = ref(localStorage.getItem("wd_agent_id") || "agent-001");
const agentConnected = ref(false);
const queue = ref<QueueItem[]>([]);
const servingUser = ref<string | null>(null);
const agentInput = ref("");
const agentMessages = ref<ChatMessage[]>([]);

let userWs: WebSocket | null = null;
let agentWs: WebSocket | null = null;

const userStatusText = computed(() => {
  if (!connected.value) return "连接中断";
  if (transferMode.value === "human") {
    return `人工服务中 · ${currentAgent.value || "坐席"}`;
  }
  if (transferMode.value === "queue") {
    return `排队中 · 第 ${queuePosition.value ?? "-"} 位`;
  }
  return "AI 服务中";
});

function wsUrl(path: string) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}${path}`;
}

function pushMessage(role: Role, text: string) {
  if (!text) return;
  messages.value.push({ role, text });
  void nextTick(() => {
    document.querySelector(".message-list")?.lastElementChild?.scrollIntoView({
      behavior: "smooth",
    });
  });
}

async function loadHistory() {
  try {
    const response = await fetch(
      `/api/chat/history?sender_id=${encodeURIComponent(senderId.value)}`,
    );
    if (!response.ok) return;
    const data = await response.json();
    messages.value = (data.messages || []).map((item: any) => ({
      role: item.role === "user" ? "user" : "bot",
      text: item.text || "",
    }));
  } catch {
    // 后端未启动时保留空历史，WebSocket 状态会提示连接失败。
  }
}

function connectUser() {
  userWs?.close();
  connected.value = false;

  userWs = new WebSocket(
    wsUrl(`/ws/user/${encodeURIComponent(senderId.value)}`),
  );

  userWs.onopen = () => {
    connected.value = true;
  };

  userWs.onclose = () => {
    connected.value = false;
    transferMode.value = "machine";
  };

  userWs.onmessage = (event) => {
    const payload = JSON.parse(event.data);

    if (payload.type === "bot_messages") {
      for (const item of payload.messages || []) {
        pushMessage("bot", item.text || "");
      }
      return;
    }

    if (payload.type === "queue_position") {
      transferMode.value = "queue";
      queuePosition.value = payload.position ?? null;
      return;
    }

    if (payload.type === "transfer_accepted") {
      transferMode.value = "human";
      currentAgent.value = payload.agent_id || null;
      queuePosition.value = null;
      pushMessage("system", "人工客服已接入，后续消息将直接发送给坐席。");
      return;
    }

    if (payload.type === "agent_message") {
      pushMessage("agent", payload.text || "");
      return;
    }

    if (payload.type === "transfer_cancelled") {
      transferMode.value = "machine";
      queuePosition.value = null;
      pushMessage("system", "已取消人工排队，继续由智能客服处理。");
      return;
    }

    if (payload.type === "session_closed") {
      transferMode.value = "machine";
      currentAgent.value = null;
      pushMessage("system", "人工会话已结束，已切回智能客服。");
      return;
    }

    if (payload.type === "error") {
      pushMessage("system", payload.detail || "请求失败");
    }
  };
}

function sendUserMessage() {
  const text = input.value.trim();
  if (!text || !userWs || userWs.readyState !== WebSocket.OPEN) return;

  pushMessage("user", text);
  userWs.send(JSON.stringify({ type: "chat", text }));
  input.value = "";
}

function requestTransfer() {
  if (!userWs || userWs.readyState !== WebSocket.OPEN) return;
  userWs.send(JSON.stringify({ type: "transfer_request" }));
}

function cancelTransfer() {
  if (!userWs || userWs.readyState !== WebSocket.OPEN) return;
  userWs.send(JSON.stringify({ type: "cancel_transfer" }));
}

function connectAgent() {
  localStorage.setItem("wd_agent_id", agentId.value);
  agentWs?.close();
  agentConnected.value = false;

  agentWs = new WebSocket(
    wsUrl(`/ws/agent/${encodeURIComponent(agentId.value)}`),
  );

  agentWs.onopen = () => {
    agentConnected.value = true;
  };

  agentWs.onclose = () => {
    agentConnected.value = false;
    servingUser.value = null;
  };

  agentWs.onmessage = (event) => {
    const payload = JSON.parse(event.data);

    if (payload.type === "agent_ready" || payload.type === "queue_update") {
      queue.value = payload.queue || [];
      return;
    }

    if (payload.type === "transfer_accepted") {
      servingUser.value = payload.sender_id;
      agentMessages.value.push({
        role: "system",
        text: `已接入用户 ${payload.sender_id}`,
      });
      return;
    }

    if (payload.type === "user_message") {
      servingUser.value = payload.sender_id;
      agentMessages.value.push({
        role: "user",
        text: payload.text || "",
      });
      return;
    }

    if (payload.type === "session_closed") {
      agentMessages.value.push({
        role: "system",
        text: `会话已结束：${payload.reason || "closed"}`,
      });
      servingUser.value = null;
      return;
    }

    if (payload.type === "error") {
      agentMessages.value.push({
        role: "system",
        text: payload.detail || "坐席操作失败",
      });
    }
  };
}

function acceptUser(sender: string) {
  if (!agentWs || agentWs.readyState !== WebSocket.OPEN) return;
  agentWs.send(JSON.stringify({ type: "accept", sender_id: sender }));
}

function sendAgentMessage() {
  const text = agentInput.value.trim();
  if (
    !text ||
    !servingUser.value ||
    !agentWs ||
    agentWs.readyState !== WebSocket.OPEN
  ) {
    return;
  }

  agentMessages.value.push({ role: "agent", text });
  agentWs.send(
    JSON.stringify({
      type: "agent_chat",
      sender_id: servingUser.value,
      text,
    }),
  );
  agentInput.value = "";
}

function closeAgentSession() {
  if (
    !servingUser.value ||
    !agentWs ||
    agentWs.readyState !== WebSocket.OPEN
  ) {
    return;
  }
  agentWs.send(
    JSON.stringify({
      type: "close_session",
      sender_id: servingUser.value,
    }),
  );
}

onMounted(async () => {
  await loadHistory();
  connectUser();
});

onBeforeUnmount(() => {
  userWs?.close();
  agentWs?.close();
});
</script>

<template>
  <main class="shell">
    <header class="topbar">
      <div>
        <div class="eyebrow">WANGDIANTONG · AI CUSTOMER SERVICE</div>
        <h1>旺店通智能客服</h1>
      </div>

      <div class="mode-switch">
        <button :class="{ active: mode === 'user' }" @click="mode = 'user'">
          用户端
        </button>
        <button :class="{ active: mode === 'agent' }" @click="mode = 'agent'">
          坐席端
        </button>
      </div>
    </header>

    <section v-if="mode === 'user'" class="workspace">
      <aside class="sidebar">
        <div class="panel">
          <div class="panel-title">当前会话</div>
          <label>sender_id</label>
          <input v-model="senderId" @change="connectUser" />
          <div class="status" :data-mode="transferMode">
            <span class="dot"></span>
            {{ userStatusText }}
          </div>
        </div>

        <div class="panel">
          <div class="panel-title">业务能力</div>
          <ul class="capabilities">
            <li>多平台订单状态查询</li>
            <li>订单物流查询</li>
            <li>售后 / 退款建议</li>
            <li>相似商品推荐</li>
            <li>人工客服转接</li>
          </ul>
        </div>

        <div class="panel safety">
          <div class="panel-title">安全边界</div>
          <p>AI 只生成 TurnPlan 和处理建议，不直接退款、取消订单或修改审批状态。</p>
        </div>
      </aside>

      <section class="chat-card">
        <div class="chat-head">
          <div>
            <strong>客服对话</strong>
            <span>{{ connected ? "WebSocket 已连接" : "WebSocket 未连接" }}</span>
          </div>
          <div class="transfer-actions">
            <button
              v-if="transferMode === 'machine'"
              class="secondary"
              @click="requestTransfer"
            >
              转人工
            </button>
            <button
              v-else-if="transferMode === 'queue'"
              class="secondary"
              @click="cancelTransfer"
            >
              取消排队
            </button>
          </div>
        </div>

        <div class="message-list">
          <div v-if="messages.length === 0" class="empty">
            <strong>可以直接输入：</strong>
            <span>“查订单 WD2026090201 的物流”</span>
            <span>“这个订单商品破损，给我退款处理建议”</span>
            <span>“我要转人工”</span>
          </div>

          <div
            v-for="(item, index) in messages"
            :key="index"
            class="message"
            :class="item.role"
          >
            <div class="role">
              {{
                item.role === "user"
                  ? "用户"
                  : item.role === "agent"
                    ? "人工客服"
                    : item.role === "system"
                      ? "系统"
                      : "AI"
              }}
            </div>
            <div class="bubble">{{ item.text }}</div>
          </div>
        </div>

        <form class="composer" @submit.prevent="sendUserMessage">
          <textarea
            v-model="input"
            rows="2"
            placeholder="输入订单、物流、售后或商品问题…"
            @keydown.enter.exact.prevent="sendUserMessage"
          />
          <button type="submit" :disabled="!connected">发送</button>
        </form>
      </section>
    </section>

    <section v-else class="agent-workspace">
      <aside class="sidebar">
        <div class="panel">
          <div class="panel-title">坐席连接</div>
          <label>agent_id</label>
          <input v-model="agentId" />
          <button class="full" @click="connectAgent">
            {{ agentConnected ? "重新连接" : "上线" }}
          </button>
          <div class="status">
            <span class="dot"></span>
            {{ agentConnected ? "坐席在线" : "坐席离线" }}
          </div>
        </div>

        <div class="panel">
          <div class="panel-title">等待队列 · {{ queue.length }}</div>
          <div v-if="queue.length === 0" class="muted">暂无排队用户</div>
          <button
            v-for="item in queue"
            :key="item.sender_id"
            class="queue-item"
            @click="acceptUser(item.sender_id)"
          >
            <span>{{ item.sender_id }}</span>
            <b>#{{ item.position }}</b>
          </button>
        </div>
      </aside>

      <section class="chat-card">
        <div class="chat-head">
          <div>
            <strong>人工坐席工作台</strong>
            <span>
              {{
                servingUser
                  ? `当前服务：${servingUser}`
                  : "从左侧队列接入用户"
              }}
            </span>
          </div>
          <button
            v-if="servingUser"
            class="danger"
            @click="closeAgentSession"
          >
            结束人工会话
          </button>
        </div>

        <div class="message-list">
          <div v-if="agentMessages.length === 0" class="empty">
            接入排队用户后，这里显示用户消息与人工回复。
          </div>
          <div
            v-for="(item, index) in agentMessages"
            :key="index"
            class="message"
            :class="item.role"
          >
            <div class="role">
              {{
                item.role === "user"
                  ? "用户"
                  : item.role === "agent"
                    ? "坐席"
                    : "系统"
              }}
            </div>
            <div class="bubble">{{ item.text }}</div>
          </div>
        </div>

        <form class="composer" @submit.prevent="sendAgentMessage">
          <textarea
            v-model="agentInput"
            rows="2"
            placeholder="输入人工回复…"
            :disabled="!servingUser"
            @keydown.enter.exact.prevent="sendAgentMessage"
          />
          <button type="submit" :disabled="!servingUser">发送</button>
        </form>
      </section>
    </section>
  </main>
</template>
