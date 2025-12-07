/**
 * DOM and HTML utilities
 */

import { infoLogger } from './logger'

/**
 * Calculate total height of DOM element children
 */
export function calculateTotalChildrenHeight(parentElement: HTMLElement | null): number {
  let totalHeight = 0

  if (!parentElement || !parentElement.children) {
    return 0
  }

  for (const child of parentElement.children) {
    const style = window.getComputedStyle(child)

    const height = Number.parseFloat(style.height)
    const marginTop = Number.parseFloat(style.marginTop)
    const marginBottom = Number.parseFloat(style.marginBottom)

    totalHeight += height + marginTop + marginBottom
  }

  return Math.ceil(totalHeight)
}

interface LoadScriptResult {
  status: boolean
  message?: string
}

/**
 * Dynamically load a script
 */
export const loadScript = (
  FILE_URL: string,
  async = true,
  type = 'text/javascript',
): Promise<LoadScriptResult> => {
  return new Promise((resolve, reject) => {
    try {
      // Check if the script already exists
      let scriptEle = document.querySelector(`script[src="${FILE_URL}"]`) as HTMLScriptElement | null

      if (scriptEle) {
        scriptEle.addEventListener('load', () => {
          resolve({ status: true })
        })
        return
      }

      scriptEle = document.createElement('script')
      scriptEle.type = type
      scriptEle.async = async
      scriptEle.src = FILE_URL

      scriptEle.addEventListener('load', () => {
        resolve({ status: true })
      })

      scriptEle.addEventListener('error', () => {
        reject({
          status: false,
          message: `Failed to load the script ${FILE_URL}`,
        })
      })

      document.body.appendChild(scriptEle)
    } catch (error) {
      reject(error)
    } finally {
      infoLogger(`Finally loaded script: ${FILE_URL}`)
    }
  })
}
