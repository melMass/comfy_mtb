/**
 * Local storage management with namespacing
 */

export class LocalStorageManager {
  private namespace: string

  constructor(namespace: string) {
    this.namespace = namespace
  }

  private _namespacedKey(key: string): string {
    return `${this.namespace}:${key}`
  }

  set<T>(key: string, value: T): void {
    const serializedValue = JSON.stringify(value)
    localStorage.setItem(this._namespacedKey(key), serializedValue)
  }

  get<T>(key: string, default_val: T | null = null): T | null {
    const value = localStorage.getItem(this._namespacedKey(key))
    return value ? (JSON.parse(value) as T) : default_val
  }

  remove(key: string): void {
    localStorage.removeItem(this._namespacedKey(key))
  }

  clear(): void {
    const prefix = `${this.namespace}:`
    const keysToRemove = Object.keys(localStorage).filter((k) => k.startsWith(prefix))
    for (const key of keysToRemove) {
      localStorage.removeItem(key)
    }
  }
}
