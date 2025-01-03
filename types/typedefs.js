/**
 * @typedef {import("./shared.d.ts").NodeData} NodeData
 * @typedef {import("./shared.d.ts").NodeType} NodeType
 * @typedef {import("./shared.d.ts").DocumentationOptions} DocumentationOptions
 * @typedef {import("./shared.d.ts").OnDrawForegroundParams} OnDrawForegroundParams
 * @typedef {import("./shared.d.ts").OnMouseDownParams} OnMouseDownParams
 * @typedef {import("./shared.d.ts").OnConnectionsChangeParams} OnConnectionsChangeParams
 * @typedef {import("./shared.d.ts").ContextMenuItem} ContextMenuItem
 * @typedef {import("./shared.d.ts").IWidget} IWidget
 * @typedef {import("./shared.d.ts").VectorWidget} VectorWidget
 * @typedef {import("./shared.d.ts").LGraphNodeExtended} LGraphNode
 * @typedef {import("./shared.d.ts").LLink} LLink
 * @typedef {import("./shared.d.ts").App} App
 * @typedef {import("./shared.d.ts").OnDrawWidgetParams} OnDrawWidgetParams
 * @typedef {import("./shared.d.ts").INodeInputSlot} INodeInputSlot
 * @typedef {import("./shared.d.ts").INodeOutputSlot} INodeOutputSlot
 */

/**
 * @typedef {Object} ResultItem
 * @property {string} [filename] - The filename of the item.
 * @property {string} [subfolder] - The subfolder of the item.
 * @property {string} [type] - The type of the item.
 */

/**
 * @typedef {Object} Outputs
 * @property {ResultItem[]} [audio] - Audio result items.
 * @property {ResultItem[]} [images] - Image result items.
 * @property {ResultItem[]} [animated] - Animated result items.
 */

/**
 * @typedef {Record<string, Outputs>} TaskOutput
 * - A record mapping Node IDs to their Outputs.
 */

/**
 * @typedef {Array} TaskPrompt
 * @property {QueueIndex} [0] - The queue index.
 * @property {PromptId} [1] - The unique prompt ID.
 * @property {PromptInputs} [2] - The prompt inputs.
 * @property {ExtraData} [3] - Extra data.
 * @property {OutputsToExecute} [4] - The outputs to execute.
 */

/**
 * @typedef {Object} HistoryTaskItem
 * @property {'History'} taskType - The type of task.
 * @property {TaskPrompt} prompt - The task prompt.
 * @property {Status} [status] - The status of the task.
 * @property {TaskOutput} outputs - The task outputs.
 * @property {TaskMeta} [meta] - Optional task metadata.
 */

/**
 * @typedef {Object} ExecInfo
 * @property {number} queue_remaining - The number of items remaining in the queue.
 */

/**
 * @typedef {Object} StatusWsMessageStatus
 * @property {ExecInfo} exec_info - Execution information.
 */

