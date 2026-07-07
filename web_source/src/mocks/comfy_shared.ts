// src/mocks/comfy_shared.ts
// Mock for comfy_shared.js utilities

export function deepMerge(target: object, ...sources: object[]): object {
  for (const source of sources) {
    for (const key in source) {
      if (Object.prototype.hasOwnProperty.call(source, key)) {
        const sourceValue = (source as Record<string, unknown>)[key]
        const targetValue = (target as Record<string, unknown>)[key]
        if (
          sourceValue &&
          typeof sourceValue === 'object' &&
          !Array.isArray(sourceValue)
        ) {
          ;(target as Record<string, unknown>)[key] = deepMerge(
            (targetValue as object) || {},
            sourceValue as object,
          )
        } else {
          ;(target as Record<string, unknown>)[key] = sourceValue
        }
      }
    }
  }
  return target
}

export function addMenuHandler(
  nodeType: unknown,
  callback: (...args: unknown[]) => void,
): void {
  console.log('Mock addMenuHandler called')
}

export function extendPrototype(
  prototype: object,
  methodName: string,
  callback: (...args: unknown[]) => void,
): void {
  console.log(`Mock extendPrototype: ${methodName}`)
}

export function getNodes(includeAll = false): unknown[] {
  console.log('Mock getNodes called')
  return []
}

export function infoLogger(...args: unknown[]): void {
  console.log('[MTB Info]', ...args)
}

export function warnLogger(...args: unknown[]): void {
  console.warn('[MTB Warn]', ...args)
}

export function errorLogger(...args: unknown[]): void {
  console.error('[MTB Error]', ...args)
}
