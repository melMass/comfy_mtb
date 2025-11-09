<script lang="ts">
  import { onMount } from 'svelte'
  import { inComfy } from './utils'
  import type { NodeId } from '@comfyorg/litegraph'
  import type { ComfyApi } from '@comfyorg/comfyui-frontend-types'

  let nodeId: NodeId | undefined = $state()
  let img:string|undefined = $state()

  const show = (src:string, node?:NodeId|null) => {
    img = src
    nodeId = Number(node)
  }
  let api:ComfyApi

  onMount(() => {
    if (!inComfy()) return
    // const modname = '/scripts/api.js'
    // import(/* @vite-ignore */ modname)
    //   .then(({ api }) => {
    import("@/scripts/api").then((e)=>{
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
      // })
      // .catch((error) => {
      //   console.error(error)
      // })

    })
  })
</script>

<button
  onclick={(e) => {
    if (!inComfy()) {
      return
    }
    e.stopPropagation()
    e.preventDefault()
    const node = window.app?.graph.getNodeById(nodeId)
    if (!node) return
    window.app?.canvas.centerOnNode(node)
    window.app?.canvas.setZoom(1)
  }}
>
  {#if img}
    <img src={img} alt="preview" />
  {/if}
</button>

<style>
  button,
  img {
    width: 100%;
    height: 320px;
  }
  img {
    object-fit: contain;
  }
</style>
