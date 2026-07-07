<script lang="ts">
  import { onMount } from 'svelte'
  import { inComfy } from './utils'
  import type { NodeId } from '@comfyorg/litegraph'
  import type { ComfyApi } from '@comfyorg/comfyui-frontend-types'

  let nodeId: NodeId | undefined = $state()
  let img: string | undefined = $state()

  const show = (src: string, node?: NodeId | null) => {
    img = src
    nodeId = Number(node)
  }
  let api: ComfyApi

  onMount(() => {
    if (!inComfy()) return
    import('@/scripts/api').then((e) => {
      api = e.api
      api.addEventListener('executed', ({ detail }) => {
        const images = detail?.output?.images
        if (!images || !images.length) return
        const format = window.app?.getPreviewFormatParam()
        const src = [
          `./view?filename=${encodeURIComponent(images[0].filename)}`,
          `type=${images[0].type}`,
          `subfolder=${encodeURIComponent(images[0].subfolder)}`,
          `t=${+new Date()}${format}`,
        ].join('&')
        show(src, detail.node)
      })

      api.addEventListener('b_preview', ({ detail }) => {
        show(URL.createObjectURL(detail), window.app?.runningNodeId)
      })
    })
  })
</script>

<button
  class="mtb-preview"
  onclick={(e) => {
    if (!inComfy()) return
    e.stopPropagation()
    e.preventDefault()
    const node = window.app?.graph.getNodeById(nodeId)
    if (!node) return
    window.app?.canvas.centerOnNode(node)
    window.app?.canvas.setZoom(1)
  }}
>
  {#if img}
    <img src={img} alt="Preview output" />
  {:else}
    <div class="mtb-preview-empty">
      <span class="mtb-preview-icon">🖼</span>
      <span class="mtb-preview-text">No preview yet</span>
    </div>
  {/if}
</button>

<style>
  .mtb-preview {
    width: 100%;
    min-height: 180px;
    max-height: 320px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.3);
    border: none;
    border-bottom: 1px solid var(--border-color, rgba(255,255,255,0.06));
    cursor: pointer;
    transition: background 0.15s ease;
    padding: 0;
    margin: 0;
  }

  .mtb-preview:hover {
    background: rgba(0, 0, 0, 0.4);
  }

  .mtb-preview img {
    width: 100%;
    height: 100%;
    max-height: 320px;
    object-fit: contain;
  }

  .mtb-preview-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 40px;
  }

  .mtb-preview-icon {
    font-size: 32px;
    opacity: 0.3;
  }

  .mtb-preview-text {
    font-size: 12px;
    color: var(--descrip-text, rgba(255,255,255,0.3));
  }
</style>
