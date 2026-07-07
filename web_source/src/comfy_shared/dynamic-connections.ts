/**
 * Dynamic connections management for nodes
 */

import type { INodeInputSlot, INodeOutputSlot } from '@comfyorg/litegraph'
import { app } from '@/scripts/app'
import type { MTBNode, NodeType, DynamicConnectionOptions, ContextMenuItem } from './types'
import { infoLogger, errorLogger } from './logger'
import { nodesFromLink } from './widgets'

type SlotCondition = (slot: INodeInputSlot | INodeOutputSlot) => boolean

const isDynamicInput = (input: INodeInputSlot & { _isDynamic?: boolean }): boolean => {
  return input._isDynamic === true
}

const addDynamicInput = (
  node: MTBNode,
  name: string,
  kind: string,
): INodeInputSlot & { _isDynamic?: boolean } => {
  const input = node.addInput(name, kind) as INodeInputSlot & { _isDynamic?: boolean }
  input._isDynamic = true

  update_dynamic_properties(node)
  set_slot_colors(node, ['cyan', undefined], isDynamicInput as SlotCondition)

  return input
}

const set_slot_colors = (
  node: MTBNode,
  colors: [string | undefined, string | undefined],
  condition?: SlotCondition,
): void => {
  const check = condition || (() => true)

  for (const slot of node.slots || []) {
    if (check(slot)) {
      ;(slot as INodeInputSlot & { color_off?: string; color_on?: string }).color_off = colors[0]
      ;(slot as INodeInputSlot & { color_on?: string }).color_on = colors[1]
    }
  }
}

const update_dynamic_properties = (node: MTBNode): void => {
  const dyn: string[] = []
  for (const input of node.inputs) {
    if (isDynamicInput(input)) {
      dyn.push(input.name)
    }
  }
  node.setProperty('dynamic_connections', dyn)
}

/**
 * Setup dynamic connections for a node type
 */
export const setupDynamicConnections = (
  nodeType: NodeType,
  prefix: string,
  inputType: string | string[],
  opts?: Partial<DynamicConnectionOptions>,
): void => {
  infoLogger(
    'Setting up dynamic connections for',
    (Object.getOwnPropertyDescriptor(nodeType, 'title')?.value as string) || 'unknown',
  )

  const options: DynamicConnectionOptions = {
    separator: '_',
    start_index: 1,
    rename_menu: 'label',
    ...opts,
  }

  const is_valid_name = (_node: MTBNode, _val: string): boolean => {
    return true
  }

  nodeType.prototype.getSlotMenuOptions = (slot): ContextMenuItem[] | undefined => {
    if (!slot.input) {
      return undefined
    }
    infoLogger('Slot Menu', { slot })
    return [
      {
        content: `Rename Input (${options.rename_menu})`,
        callback: () => {
          const dialog = app.canvas.createDialog(
            "<span class='name'>Name</span><input autofocus type='text'/><button>OK</button>",
            {},
          ) as HTMLElement & { close: () => void }

          const dialogInput = dialog.querySelector('input') as HTMLInputElement | null
          if (dialogInput) {
            if (options.rename_menu === 'label') {
              dialogInput.value = slot.input!.label || slot.input!.name || ''
            } else if (options.rename_menu === 'name') {
              dialogInput.value = slot.input!.name || ''
            }
          }

          const inner = (): void => {
            const val = dialogInput?.value || ''
            if (!is_valid_name(slot.node, val)) {
              dialog.close()
              return
            }

            app.graph.beforeChange()
            if (options.rename_menu === 'label') {
              slot.input!.label = val
            } else if (options.rename_menu === 'name') {
              slot.input!.name = val
              slot.input!.label = val
            }

            app.graph.afterChange()
            dialog.close()
          }

          dialog.querySelector('button')?.addEventListener('click', inner)
          dialogInput?.addEventListener('keydown', (e: KeyboardEvent) => {
            ;(dialog as HTMLElement & { is_modified?: boolean }).is_modified = true
            if (e.keyCode === 27) {
              dialog.close()
            } else if (e.keyCode === 13) {
              inner()
            } else if (
              e.keyCode !== 13 &&
              (e.target as HTMLElement)?.localName !== 'textarea'
            ) {
              return
            }
            e.preventDefault()
            e.stopPropagation()
          })
          dialogInput?.focus()
        },
      },
    ]
  }

  const onConfigure = nodeType.prototype.onConfigure

  nodeType.prototype.onConfigure = function (this: MTBNode, data: unknown) {
    const r = onConfigure ? onConfigure.apply(this, [data]) : undefined

    if (!('dynamic_connections' in this.properties)) {
      this.setProperty('dynamic_connections', [])
    } else {
      const dynamic_connections = this.properties.dynamic_connections as string[] | string
      if (typeof dynamic_connections !== 'object') {
        return r
      }
      for (const name of dynamic_connections) {
        infoLogger(`Would dynamize: ${name}`)
        const input = this.inputs.find((i) => i.name === name)
        if (input) {
          infoLogger('Input found', { input })
          input._isDynamic = true
        }
      }
    }
    set_slot_colors(this, ['cyan', undefined], isDynamicInput as SlotCondition)

    return r
  }

  const onNodeCreated = nodeType.prototype.onNodeCreated
  const inputList = typeof inputType === 'object'

  nodeType.prototype.onNodeCreated = function (this: MTBNode) {
    const r = onNodeCreated ? onNodeCreated.apply(this) : undefined

    addDynamicInput(
      this,
      `${prefix}${options.separator}${options.start_index}`,
      inputList ? '*' : (inputType as string),
    )
    return r
  }

  const onConnectionsChange = nodeType.prototype.onConnectionsChange

  nodeType.prototype.onConnectionsChange = function (this: MTBNode, ...args) {
    const [type, slotIndex, isConnected, link, ioSlot] = args

    options.link = link
    options.ioSlot = ioSlot
    const r = onConnectionsChange
      ? onConnectionsChange.apply(this, [type, slotIndex, isConnected, link, ioSlot])
      : undefined

    options.DEBUG = {
      node: this,
      type,
      slotIndex,
      isConnected,
      link,
      ioSlot,
    }

    dynamic_connection(
      this,
      slotIndex,
      isConnected,
      `${prefix}${options.separator}`,
      inputType,
      options,
    )
    return r
  }
}

/**
 * Main logic around dynamic inputs
 */
export const dynamic_connection = (
  node: MTBNode,
  index: number,
  connected: boolean,
  connectionPrefix = 'input_',
  connectionType: string | string[] = '*',
  opts?: Partial<DynamicConnectionOptions>,
): void => {
  const options: DynamicConnectionOptions = {
    start_index: 1,
    ...opts,
  }

  if (node.inputs.length > 0 && !isDynamicInput(node.inputs[index])) {
    return
  }

  const listConnection = typeof connectionType === 'object'
  const conType = listConnection ? '*' : connectionType
  const nameArray = options.nameArray || []

  const clean_inputs = (): void => {
    if (node.id < 0) return // being duplicated
    if (node.inputs.length === 0) return

    let w_count = node.widgets?.length || 0
    let i_count = node.inputs?.length || 0
    infoLogger(`Cleaning inputs: [BEFORE] (w: ${w_count} | inputs: ${i_count})`)

    const to_remove: number[] = []
    for (let n = 1; n < node.inputs.length; n++) {
      const element = node.inputs[n]
      if (!element.link && isDynamicInput(element)) {
        if (node.widgets) {
          const w = node.widgets.find((w) => w.name === element.name)
          if (w) {
            w.onRemoved?.()
            node.widgets.length = node.widgets.length - 1
          }
        }
        infoLogger(`Removing input ${n}`)
        to_remove.push(n)
      }
    }

    for (let i = 0; i < to_remove.length; i++) {
      const id = to_remove[i]
      try {
        node.removeInput(id)
        i_count -= 1
      } catch (err) {
        errorLogger('Cannot remove input', err)
      }
    }
    node.inputs.length = i_count

    w_count = node.widgets?.length || 0
    i_count = node.inputs?.length || 0
    infoLogger(`Cleaning inputs: [AFTER] (w: ${w_count} | inputs: ${i_count})`)

    infoLogger('Cleaning inputs: making it sequential again')
    // make inputs sequential again
    let prefixed_idx = options.start_index!
    for (let i = 0; i < node.inputs.length; i++) {
      let name = ''
      if (node.inputs[i].name.startsWith(connectionPrefix)) {
        name = `${connectionPrefix}${prefixed_idx}`
        prefixed_idx += 1
      } else {
        name = node.inputs[i].name
      }

      if (nameArray.length > 0) {
        name = i < nameArray.length ? nameArray[i] : name
      }

      // preserve label if it exists
      ;(node.inputs[i] as INodeInputSlot & { label?: string }).label =
        (node.inputs[i] as INodeInputSlot & { label?: string }).label || name
      node.inputs[i].name = name
    }
  }

  if (!connected) {
    if (!options.link) {
      infoLogger('Disconnecting', { options })
      clean_inputs()
    } else {
      if (!options.ioSlot?.link) {
        node.connectionTransit = true
      } else {
        node.connectionTransit = false
        clean_inputs()
      }
      infoLogger('Reconnecting', { options })
    }
  }

  if (connected) {
    if (options.link) {
      const { from, to, type } = nodesFromLink(node, options.link)
      if (type === 'outgoing') return
      infoLogger('Connecting', { options, from, to, type })
    } else {
      infoLogger('Connecting', { options })
    }

    if (node.connectionTransit) {
      infoLogger('In Transit')
      node.connectionTransit = false
    }

    clean_inputs()

    if (node.inputs.length === 0) return

    // add an extra input
    if (node.inputs[node.inputs.length - 1].link !== null) {
      const nextIndex = node.inputs.reduce(
        (acc, cur) => (isDynamicInput(cur) ? ++acc : acc),
        0,
      )

      const name =
        nextIndex < nameArray.length
          ? nameArray[nextIndex]
          : `${connectionPrefix}${nextIndex + options.start_index!}`

      infoLogger(`Adding input ${nextIndex + 1} (${name})`)
      addDynamicInput(node, name, conType as string)
    }
  }
}
