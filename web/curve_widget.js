// Reference the shared typedefs file
/// <reference path="../types/typedefs.js" />
import { app } from '../../scripts/app.js'
import { infoLogger } from './comfy_shared.js'

function B0(t) {
  return (1 - t) ** 3 / 6
}
function B1(t) {
  return (3 * t ** 3 - 6 * t ** 2 + 4) / 6
}
function B2(t) {
  return (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6
}
function B3(t) {
  return t ** 3 / 6
}
class CurveWidget {
  constructor(...args) {
    const [inputName, opts] = args

    this.name = inputName || 'Curve'

    this.type = 'FLOAT_CURVE'
    this.selectedPointIndex = null
    this.options = opts
    this.value = this.value || { 0: { x: 0, y: 0 }, 1: { x: 1, y: 1 } }
  }

  drawBSpline(ctx, width, height, posY) {
    const n = this.value.length - 1
    const numSegments = n - 2
    const numPoints = this.value.length
    if (numPoints < 4) {
      this.drawLinear(ctx, width, height, posY)
    } else {
      for (let j = 0; j <= numSegments; j++) {
        for (let t = 0; t <= 1; t += 0.01) {
          let pt = this.getBSplinePoint(j, t)
          let x = pt.x * width
          let y = posY + height - pt.y * height

          if (t === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
      }
      ctx.stroke()
    }
  }

  drawLinear(ctx, width, height, posY) {
    for (let i = 0; i < Object.keys(this.value).length - 1; i++) {
      let p1 = this.value[i]
      let p2 = this.value[i + 1]
      ctx.moveTo(p1.x * width, posY + height - p1.y * height)
      ctx.lineTo(p2.x * width, posY + height - p2.y * height)
    }
    ctx.stroke()
  }
  getBSplinePoint(i, t) {
    // Control points for this segment
    const p0 = this.value[i]
    const p1 = this.value[i + 1]
    const p2 = this.value[i + 2]
    const p3 = this.value[i + 3]

    const x = B0(t) * p0.x + B1(t) * p1.x + B2(t) * p2.x + B3(t) * p3.x
    const y = B0(t) * p0.y + B1(t) * p1.y + B2(t) * p2.y + B3(t) * p3.y

    return { x, y }
  }
  /**
   * @param {OnDrawWidgetParams} args
   */
  draw(...args) {
    const hide = this.type !== 'FLOAT_CURVE'
    if (hide) {
      return
    }

    const [ctx, node, width, posY, height] = args
    const [cw, ch] = this.computeSize(width)

    ctx.beginPath()
    ctx.fillStyle = '#000'
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2

    // normalized coordinates -> canvas coordinates
    for (let i = 0; i < Object.keys(this.value || {}).length - 1; i++) {
      let p1 = this.value[i]
      let p2 = this.value[i + 1]
      ctx.moveTo(p1.x * cw, posY + ch - p1.y * ch)
      ctx.lineTo(p2.x * cw, posY + ch - p2.y * ch)
    }
    ctx.stroke()

    // points
    Object.values(this.value || {}).forEach((point) => {
      ctx.beginPath()
      ctx.arc(point.x * cw, posY + ch - point.y * ch, 5, 0, 2 * Math.PI)
      ctx.fill()
    })
  }

  mouse(event, pos, node) {
    let x = pos[0] - node.pos[0]
    let y = pos[1] - node.pos[1]
    const width = node.size[0]
    const height = 300 // TODO: compute
    const posY = node.pos[1]
    const localPos = { x: pos[0], y: pos[1] - LiteGraph.NODE_WIDGET_HEIGHT }

    if (event.type === LiteGraph.pointerevents_method + 'down') {
      console.debug('Checking if a point was clicked')
      const clickedPointIndex = this.detectPoint(localPos, width, height)
      if (clickedPointIndex !== null) {
        this.selectedPointIndex = clickedPointIndex
      } else {
        this.addPoint(localPos, width, height)
      }
      return true
    } else if (
      event.type === LiteGraph.pointerevents_method + 'move' &&
      this.selectedPointIndex !== null
    ) {
      this.movePoint(this.selectedPointIndex, localPos, width, height)
      return true
    } else if (
      event.type === LiteGraph.pointerevents_method + 'up' &&
      this.selectedPointIndex !== null
    ) {
      this.selectedPointIndex = null
      return true
    }
    return false
  }
  callback(...args) {
    //value, that, node, pos, event) {

  }

  detectPoint(localPos, width, height) {
    const threshold = 20 // TODO: extract
    const keys = Object.keys(this.value)
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i]
      const p = this.value[key]
      const px = p.x * width
      const py = height - p.y * height
      if (
        Math.abs(localPos.x - px) < threshold &&
        Math.abs(localPos.y - py) < threshold
      ) {
        return key
      }
    }
    return null
  }
  addPoint(localPos, width, height) {
    // add a new point based on click position
    const normalizedPoint = {
      x: localPos.x / width,
      y: 1 - localPos.y / height,
    }

    const keys = Object.keys(this.value)
    let insertIndex = keys.length
    for (let i = 0; i < keys.length; i++) {
      if (normalizedPoint.x < this.value[keys[i]].x) {
        insertIndex = i
        break
      }
    }
    // shift
    for (let i = keys.length; i > insertIndex; i--) {
      this.value[i] = this.value[i - 1]
    }

    this.value[insertIndex] = normalizedPoint
  }

  movePoint(index, localPos, width, height) {
    const point = this.value[index]
    point.x = Math.max(0, Math.min(1, localPos.x / width))
    point.y = Math.max(0, Math.min(1, 1 - localPos.y / height))

    this.value[index] = point
  }
  computeSize(width) {
    return [width, 300]
  }

  configure(data) {
  }
}

app.registerExtension({
  name: 'mtb.curves',
  getCustomWidgets: () => {
    return {
      /**
       * @param {LGraphNode} node
       * @param {str} inputName
       * @param {[str,*]} inputData
       * @param {*} app
       *
       */
      FLOAT_CURVE: (node, inputName, inputData, app) => {
        // const c = node.widgets.find((w) => w.type === "FLOAT_CURVE")
        const wid = node.addCustomWidget(new CurveWidget(inputName, inputData))

        return {
          widget: wid,
          minWidth: 150,
          minHeight: 30,
        }
      },
    }
  },
})
