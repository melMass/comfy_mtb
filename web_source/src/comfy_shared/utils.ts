/**
 * Base utilities
 */

/**
 * Computes the convex hull of a set of points using the Monotone Chain algorithm.
 */
export const getConvexHull = (points: [number, number][]): [number, number][] => {
  if (points.length <= 3) {
    return points
  }

  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1])

  const cross_product = (o: [number, number], a: [number, number], b: [number, number]) => {
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  }

  const lower: [number, number][] = []
  for (const p of sorted) {
    while (
      lower.length >= 2 &&
      cross_product(lower[lower.length - 2], lower[lower.length - 1], p) <= 0
    ) {
      lower.pop()
    }
    lower.push(p)
  }

  const upper: [number, number][] = []
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i]
    while (
      upper.length >= 2 &&
      cross_product(upper[upper.length - 2], upper[upper.length - 1], p) <= 0
    ) {
      upper.pop()
    }
    upper.push(p)
  }

  return lower.slice(0, lower.length - 1).concat(upper.slice(0, upper.length - 1))
}

/**
 * Generates a crude UUID v4
 */
export function makeUUID(): string {
  let dt = new Date().getTime()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0
    dt = Math.floor(dt / 16)
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
  return uuid
}

export interface Debounced<T extends (...args: unknown[]) => void> {
  (...args: Parameters<T>): void
  cancel: () => void
}

/**
 * Basic debounce decorator
 */
export function debounce<T extends (...args: unknown[]) => void>(
  func: T,
  delay: number,
): Debounced<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined

  const debounced = function (this: unknown, ...args: Parameters<T>) {
    clearTimeout(timeout)
    timeout = setTimeout(() => func.apply(this, args), delay)
  } as Debounced<T>

  debounced.cancel = () => {
    clearTimeout(timeout)
  }

  return debounced
}

/**
 * Deep merge two objects.
 */
export function deepMerge<T extends Record<string, unknown>>(
  target: T,
  ...sources: Partial<T>[]
): T {
  if (!sources.length) return target
  const source = sources.shift()

  if (source) {
    for (const key in source) {
      const sourceValue = source[key]
      if (sourceValue instanceof Object && !Array.isArray(sourceValue)) {
        if (!target[key]) Object.assign(target, { [key]: {} })
        deepMerge(target[key] as Record<string, unknown>, sourceValue as Record<string, unknown>)
      } else {
        Object.assign(target, { [key]: sourceValue })
      }
    }
  }

  return deepMerge(target, ...sources)
}

type Replacer = (value: unknown) => unknown

/**
 * Safely serialize an object to JSON, handling circular references
 */
export const safe_json = (object: unknown, replacer?: Replacer): unknown => {
  const objects = new WeakMap<object, string>()

  const derez = (value: unknown, path: string): unknown => {
    if (replacer !== undefined) {
      value = replacer(value)
    }

    if (
      typeof value === 'object' &&
      value !== null &&
      !(value instanceof Boolean) &&
      !(value instanceof Date) &&
      !(value instanceof Number) &&
      !(value instanceof RegExp) &&
      !(value instanceof String)
    ) {
      const old_path = objects.get(value)
      if (old_path !== undefined) {
        return { $ref: old_path }
      }
      objects.set(value, path)

      if (Array.isArray(value)) {
        const nu: unknown[] = []
        value.forEach((element, i) => {
          nu[i] = derez(element, `${path}[${i}]`)
        })
        return nu
      } else {
        const nu: Record<string, unknown> = {}
        Object.keys(value).forEach((name) => {
          nu[name] = derez(
            (value as Record<string, unknown>)[name],
            path + '[' + JSON.stringify(name) + ']',
          )
        })
        return nu
      }
    }
    return value
  }

  return derez(object, '$')
}

// Declare app global for ComfyUI
declare const app: {
  graph: {
    _nodes: import('./types').MTBNode[]
    computeExecutionOrder?: (onlyOnExecute: boolean) => import('./types').MTBNode[]
  }
}

/**
 * Get all nodes from the current graph
 * @param sorted - If true, returns nodes in execution order
 */
export function getNodes(sorted = false): import('./types').MTBNode[] {
  if (!app?.graph?._nodes) {
    return []
  }

  if (sorted) {
    // computeExecutionOrder returns nodes in dependency order
    return app.graph.computeExecutionOrder?.(false) ?? app.graph._nodes
  }

  return app.graph._nodes
}
