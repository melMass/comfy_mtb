/**
 * Node extension utilities
 */

import type { MTBNode, NodeType, ContextMenuItem } from './types'
import { errorLogger } from './logger'

/**
 * Extend an object, either replacing the original property or extending it.
 */
export function chainCallback<T extends object, K extends keyof T>(
  object: T | undefined,
  property: K,
  callback: T[K],
): void {
  if (object === undefined) {
    errorLogger('Could not extend undefined object', { object, property })
    return
  }

  if (property in object) {
    const callback_orig = object[property] as ((...args: unknown[]) => unknown) | undefined
    ;(object as Record<K, unknown>)[property] = function (
      this: unknown,
      ...args: unknown[]
    ): unknown {
      const r = callback_orig?.apply(this, args)
      const n = (callback as (...args: unknown[]) => unknown).apply(this, args)
      return r || n
    }
  } else {
    object[property] = callback
  }
}

/**
 * Appends a callback to the extra menu options of a given node type.
 */
export function addMenuHandler(
  nodeType: NodeType,
  cb: (this: MTBNode, app: unknown, options: ContextMenuItem[]) => ContextMenuItem[],
): void {
  const getOpts = nodeType.prototype.getExtraMenuOptions

  nodeType.prototype.getExtraMenuOptions = function (
    this: MTBNode,
    app: unknown,
    options: ContextMenuItem[],
  ): ContextMenuItem[] {
    const r = getOpts?.apply(this, [app, options]) || []
    const newItems = cb.apply(this, [app, options]) || []
    return [...r, ...newItems]
  }
}

/**
 * Prefixes the node title with '[DEPRECATED]' and log the deprecation reason to the console.
 */
export const addDeprecation = (nodeType: NodeType, reason: string): void => {
  const title = nodeType.title || 'Unknown'
  nodeType.title = `[DEPRECATED] ${title}`

  const styles = {
    title: 'font-size:1.3em;font-weight:900;color:yellow; background: black',
    reason: 'font-size:1.2em',
  }
  // biome-ignore lint/suspicious/noConsole: intentional deprecation warning
  console.log(`%c!  ${title} is deprecated:%c ${reason}`, styles.title, styles.reason)
}
