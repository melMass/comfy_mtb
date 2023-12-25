
import { app } from '../../scripts/app.js'

function B0(t) { return (1 - t) ** 3 / 6; }
function B1(t) { return (3 * t ** 3 - 6 * t ** 2 + 4) / 6; }
function B2(t) { return (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6; }
function B3(t) { return t ** 3 / 6; }

class CurveWidget {
    constructor(inputName, defaultValue) {
        this.name = inputName || "Curve";
        this._value = defaultValue || [{ x: 0, y: 0 }, { x: 1, y: 1 }];
        this.type = "FLOAT_CURVE";
        this.selectedPointIndex = null;
        this.resize
    }

    drawBSpline(ctx, width, height, posY) {
        const n = this._value.length - 1;
        const numSegments = n - 2;
        const numPoints = this._value.length;
        if (numPoints < 4) {
            this.drawLinear(ctx, width, height, posY);
        } else {
            for (let j = 0; j <= numSegments; j++) {
                for (let t = 0; t <= 1; t += 0.01) {
                    let pt = this.getBSplinePoint(j, t);
                    let x = pt.x * width;
                    let y = posY + height - pt.y * height;

                    if (t === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }
    }

    drawLinear(ctx, width, height, posY) {
        for (let i = 0; i < this._value.length - 1; i++) {
            let p1 = this._value[i];
            let p2 = this._value[i + 1];
            ctx.moveTo(p1.x * width, posY + height - p1.y * height);
            ctx.lineTo(p2.x * width, posY + height - p2.y * height);
        }
        ctx.stroke();
    }

    getBSplinePoint(i, t) {
        // Control points for this segment
        const p0 = this._value[i];
        const p1 = this._value[i + 1];
        const p2 = this._value[i + 2];
        const p3 = this._value[i + 3];

        const x = B0(t) * p0.x + B1(t) * p1.x + B2(t) * p2.x + B3(t) * p3.x;
        const y = B0(t) * p0.y + B1(t) * p1.y + B2(t) * p2.y + B3(t) * p3.y;

        return { x, y };
    }

    draw(ctx, node, width, posY, height) {
        const [cw, ch] = this.computeSize(width)

        ctx.beginPath();
        ctx.fillStyle = "#000";
        //ctx.fillRect(0, posY, cw, ch);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;

        // normalized coordinates -> canvas coordinates
        for (let i = 0; i < this._value.length - 1; i++) {
            let p1 = this._value[i];
            let p2 = this._value[i + 1];
            ctx.moveTo(p1.x * cw, posY + ch - p1.y * ch);
            ctx.lineTo(p2.x * cw, posY + ch - p2.y * ch);
        }
        ctx.stroke();
        // this.drawBSpline(ctx, width, height, posY);

        // points
        this._value.forEach(point => {
            ctx.beginPath();
            ctx.arc(point.x * cw, posY + ch - point.y * ch, 5, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    mouse(event, pos, node) {
        // console.debug(event.type, pos, node)
        let x = pos[0] - node.pos[0]
        let y = pos[1] - node.pos[1]
        let width = node.size[0]
        const height = 300; // TODO: compute
        const posY = node.pos[1];

        const localPos = { x: pos[0], y: pos[1] - LiteGraph.NODE_WIDGET_HEIGHT };

        if (event.type === LiteGraph.pointerevents_method + "down") {
            console.debug("Checking if a point was clicked");
            const clickedPointIndex = this.detectPoint(localPos, width, height);
            if (clickedPointIndex !== null) {
                this.selectedPointIndex = clickedPointIndex;
            } else {
                this.addPoint(localPos, width, height);
            }
            return true;
        } else if (event.type === LiteGraph.pointerevents_method + "move" && this.selectedPointIndex !== null) {
            this.movePoint(this.selectedPointIndex, localPos, width, height);
            return true;
        } else if (event.type === LiteGraph.pointerevents_method + "up" && this.selectedPointIndex !== null) {
            this.selectedPointIndex = null;
            return true;
        }
        return false;
    }


    detectPoint(localPos, width, height) {
        const threshold = 20; // TODO: extract
        for (let i = 0; i < this._value.length; i++) {
            const p = this._value[i];
            const px = p.x * width;
            const py = height - p.y * height;
            if (Math.abs(localPos.x - px) < threshold && Math.abs(localPos.y - py) < threshold) {
                return i;
            }
        }
        return null;
    }

    addPoint(localPos, width, height) {
        // add a new point based on click position
        const normalizedPoint = { x: localPos.x / width, y: 1 - localPos.y / height };
        this._value.push(normalizedPoint);
        this._value.sort((a, b) => a.x - b.x);
        this.value = JSON.stringify(this._value);
    }

    movePoint(index, localPos, width, height) {
        const point = this._value[index];
        point.x = Math.max(0, Math.min(1, localPos.x / width));
        point.y = Math.max(0, Math.min(1, 1 - localPos.y / height));

        this._value[index] = point;
        this.value = JSON.stringify(this._value);
    }

    computeSize(width) {
        return [width, 300];
    }

    configure(data) {
        console.log(data)
    }

    value() {
        console.debug('Returning value', this._value)
        return this._value
    }
    setValue(value) {
        console.debug('Setting value', value)
        this._value = value
    }
}

app.registerExtension({
    name: 'mtb.curves',
    getCustomWidgets: function () {
        return {
            FLOAT_CURVE: (node, inputName, inputData, app) => {
                console.debug('Registering float curve widget');

                return {
                    widget: node.addCustomWidget(
                        new CurveWidget(inputName, inputData[1]?.default)
                    ),
                    minWidth: 150,
                    minHeight: 30,
                }
            },


        }
    },

})
