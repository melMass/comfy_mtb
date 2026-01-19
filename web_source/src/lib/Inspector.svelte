<script lang="ts">
  import { fade } from 'svelte/transition'
  import { resizeHandle } from './actions.js'

  import InspectorPreview from './InspectorPreview.svelte'
  import Tabs from './Tabs.svelte'
  import TabUi from './TabUI.svelte'
  import TabHelp from './TabHelp.svelte'
  import { inComfy } from './utils'
  import { onMount } from 'svelte'
  import { graphToPrompt } from '../mtb_api/graph-to-prompt'

  interface InputItem {
    id: number
    name: string
    type: string
    options?: string[]
    [key: string]: unknown
  }

  let {
    visible = true,
    inputs = {} as Record<string, InputItem>,
  } = $props()

  const items = $derived(
    Object.keys(inputs).map((k) => ({
      ...inputs[k],
      original_name: k,
    }))
  )

  onMount(() => {
    console.log('Mounted API Inspector')
  })

  async function exportApi() {
    if (!inComfy()) return

    // @ts-expect-error - app is global in ComfyUI
    const { output } = await graphToPrompt(app)

    const json = JSON.stringify(output, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)

    const a = document.createElement('a')
    a.href = url
    a.download = 'workflow_api.json'
    a.click()

    URL.revokeObjectURL(url)
  }
</script>

{#if visible}
  <div transition:fade={{ duration: 120 }} use:resizeHandle class="mtb-panel">
    <header class="mtb-panel-header">
      <div class="mtb-panel-title">
        <span class="mtb-panel-icon">&#9889;</span>
        <span>API Controls</span>
      </div>
      <span class="mtb-panel-badge">{items.length}</span>
    </header>

    <div class="mtb-panel-content">
      <InspectorPreview />

      <Tabs
        items={[
          {
            label: 'Inputs',
            value: 1,
            component: TabUi,
            props: { inputs: items },
          },
          {
            label: 'Help',
            value: 2,
            component: TabHelp,
            props: {},
          },
        ]}
      />
    </div>

    <footer class="mtb-panel-footer">
      <button class="mtb-btn mtb-btn-primary" onclick={() => {
        if (!inComfy()) return
        // @ts-expect-error - app is global in ComfyUI
        app.queuePrompt(0, 1)
      }}>
        <span class="mtb-btn-icon">&#9654;</span>
        Queue
      </button>
      <button class="mtb-btn mtb-btn-secondary" onclick={exportApi}>Export</button>
    </footer>

    <div class="mtb-panel-status">
      <span class="mtb-status-dot"></span>
      <span>Ready</span>
    </div>
  </div>
{/if}

<style>
  /* MTB Panel - Vercel-inspired minimal design */
  .mtb-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--comfy-menu-bg, #1a1a1a);
    color: var(--fg-color, #fafafa);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  /* Header */
  .mtb-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color, rgba(255,255,255,0.08));
    background: var(--comfy-menu-bg, #1a1a1a);
  }

  .mtb-panel-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 500;
    font-size: 13px;
    letter-spacing: -0.01em;
  }

  .mtb-panel-icon {
    font-size: 14px;
    opacity: 0.9;
  }

  .mtb-panel-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 20px;
    height: 20px;
    padding: 0 6px;
    background: var(--mtb-api-color, #2930b0);
    color: white;
    font-size: 11px;
    font-weight: 600;
    border-radius: 10px;
  }

  /* Content */
  .mtb-panel-content {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-width: thin;
    scrollbar-color: var(--fg-color, #888) transparent;
  }

  .mtb-panel-content::-webkit-scrollbar {
    width: 6px;
  }

  .mtb-panel-content::-webkit-scrollbar-track {
    background: transparent;
  }

  .mtb-panel-content::-webkit-scrollbar-thumb {
    background: var(--border-color, rgba(255,255,255,0.15));
    border-radius: 3px;
  }

  .mtb-panel-content::-webkit-scrollbar-thumb:hover {
    background: var(--fg-color, rgba(255,255,255,0.25));
  }

  /* Footer */
  .mtb-panel-footer {
    display: flex;
    gap: 8px;
    padding: 12px 16px;
    border-top: 1px solid var(--border-color, rgba(255,255,255,0.08));
  }

  /* Buttons */
  .mtb-btn {
    flex: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    height: 32px;
    padding: 0 12px;
    font-size: 12px;
    font-weight: 500;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .mtb-btn-primary {
    background: var(--mtb-api-color, #2930b0);
    color: white;
  }

  .mtb-btn-primary:hover {
    background: color-mix(in srgb, var(--mtb-api-color, #2930b0), white 10%);
    transform: translateY(-1px);
  }

  .mtb-btn-secondary {
    background: var(--comfy-input-bg, rgba(255,255,255,0.05));
    color: var(--fg-color, #fafafa);
    border: 1px solid var(--border-color, rgba(255,255,255,0.1));
  }

  .mtb-btn-secondary:hover {
    background: var(--border-color, rgba(255,255,255,0.1));
    border-color: var(--fg-color, rgba(255,255,255,0.2));
  }

  .mtb-btn-icon {
    font-size: 10px;
  }

  /* Status bar */
  .mtb-panel-status {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 6px 16px;
    font-size: 11px;
    color: var(--descrip-text, rgba(255,255,255,0.5));
    background: rgba(0,0,0,0.2);
  }

  .mtb-status-dot {
    width: 6px;
    height: 6px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
</style>
