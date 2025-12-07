// src/mocks/api.ts

export const api = {
  addEventListener: (event: string, callback: (e: CustomEvent) => void) => {
    console.log(`Mock api.addEventListener: ${event}`)
  },
  removeEventListener: (event: string, callback: (e: CustomEvent) => void) => {
    console.log(`Mock api.removeEventListener: ${event}`)
  },
  getItems: async (type: string) => {
    console.log(`Mock api.getItems: ${type}`)
    return { Running: [], Pending: [] }
  },
  deleteItem: async (type: string, id: string) => {
    console.log(`Mock api.deleteItem: ${type} ${id}`)
  },
  clearItems: async (type: string) => {
    console.log(`Mock api.clearItems: ${type}`)
  },
}
