/**
 * Logging utilities
 */

import { app } from '@/scripts/app'

type ConsoleMethod = 'log' | 'warn' | 'error' | 'debug'
type ToastSeverity = 'info' | 'warn' | 'error' | 'secondary'

const consoleMethodToSeverity = (method: ConsoleMethod): ToastSeverity => {
  switch (method) {
    case 'log':
      return 'info'
    case 'error':
    case 'warn':
      return method
    default:
      return 'secondary'
  }
}

interface LogResult {
  notify: (timeout?: number) => void
}

type Logger = (message: string, ...args: unknown[]) => LogResult

function createLogger(emoji: string, color: string, consoleMethod: ConsoleMethod = 'log'): Logger {
  return (message: string, ...args: unknown[]): LogResult => {
    if (window.MTB?.DEBUG) {
      // biome-ignore lint/suspicious/noConsole: logger wrapper
      console[consoleMethod](`%c${emoji} ${message}`, `color: ${color};`, ...args)
    }
    return {
      notify: (timeout = 3000) => {
        app.extensionManager?.toast?.add({
          severity: consoleMethodToSeverity(consoleMethod),
          summary: 'MTB',
          detail: `${emoji} ${message}`,
          life: timeout,
        })
      },
    }
  }
}

export const infoLogger = createLogger('ℹ️', 'yellow')
export const warnLogger = createLogger('⚠️', 'orange', 'warn')
export const errorLogger = createLogger('🔥', 'red', 'error')
export const successLogger = createLogger('✅', 'green')

export const log = (...args: unknown[]): void => {
  if (window.MTB?.DEBUG) {
    // biome-ignore lint/suspicious/noConsole: logger wrapper
    console.debug(...args)
  }
}
