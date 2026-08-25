"""
LangGraph 全量改造版：整个对话系统就是一张图

设计总纲（对照被删除的自研实现）：
- 对话历史：messages 通道（LangChain HumanMessage/AIMessage + add_messages reducer），
  替代原 Session/Turn/pending_turn 手工轮次管理
- 流程状态：active_flow + flow_step 指针 + slots，替代 TaskContext/SystemContext；
  paused_flows 替代 paused_tasks 暂停栈
- 槽位收集：collect 步骤内调用 LangGraph interrupt() 暂停整图，
  下一轮 Command(resume=用户输入) 从断点恢复 —— 替代原 system_collect_information
  过场流程 + action_listen 停止机制
- 持久化：checkpointer（按 thread_id=sender_id 保存图状态），
  替代原 DialogueState 整体 JSON 序列化落 dialogue_states 表
- 挂起意图队列：pending_intents 进图状态，随 checkpoint 自动持久化
"""
