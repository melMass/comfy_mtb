// src/mocks/ui.ts

type ElementOptions = {
  textContent?: string
  onclick?: () => void
  type?: string
  [key: string]: unknown
}

/**
 * Mock implementation of ComfyUI's $el helper
 */
export function $el(
  tag: string,
  opts?: ElementOptions | HTMLElement[],
  children?: HTMLElement[],
): HTMLElement {
  const parts = tag.split('.')
  const tagName = parts[0] || 'div'
  const className = parts.slice(1).join(' ')

  const element = document.createElement(tagName)
  if (className) {
    element.className = className
  }

  if (Array.isArray(opts)) {
    // opts is children
    opts.forEach((child) => element.appendChild(child))
  } else if (opts) {
    // opts is options
    Object.entries(opts).forEach(([key, value]) => {
      if (key === 'textContent') {
        element.textContent = value as string
      } else if (key === 'onclick') {
        element.onclick = value as () => void
      } else if (key.startsWith('on')) {
        element.addEventListener(key.slice(2).toLowerCase(), value as EventListener)
      } else {
        element.setAttribute(key, String(value))
      }
    })
  }

  if (children) {
    children.forEach((child) => element.appendChild(child))
  }

  return element
}
