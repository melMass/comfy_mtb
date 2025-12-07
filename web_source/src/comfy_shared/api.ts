/**
 * Server API utilities
 */

import { api } from '@/scripts/api'

export interface ActionResult {
  result: unknown
}

export interface ServerInfo {
  [key: string]: unknown
}

/**
 * Run a server action
 */
export const runAction = async (name: string, ...args: unknown[]): Promise<unknown> => {
  const req = await api.fetchApi('/mtb/actions', {
    method: 'POST',
    body: JSON.stringify({
      name,
      args,
    }),
  })

  const res = (await req.json()) as ActionResult
  return res.result
}

/**
 * Get server info
 */
export const getServerInfo = async (): Promise<ServerInfo> => {
  const res = await api.fetchApi('/mtb/server-info')
  return (await res.json()) as ServerInfo
}

/**
 * Set server info
 */
export const setServerInfo = async (opts: ServerInfo): Promise<void> => {
  await api.fetchApi('/mtb/server-info', {
    method: 'POST',
    body: JSON.stringify(opts),
  })
}
