import { app } from "../../scripts/app.js";

/**
 * Generic checkbox multi-select widget for GraftingRayman nodes.
 *
 * Any STRING input whose python-side config sets `multi_select: true` and
 * provides an `options` array gets its default text widget swapped for a
 * button that opens a searchable checklist popup. Selected options are kept
 * as a comma-separated string in widget.value, so the node still works from
 * the API with a plain comma-separated string and nothing changes on the
 * python side beyond reading that string.
 */

const NODE_NAME = "GRPromptReplacerAttributesMulti";

function parseCsv(value) {
    if (!value) return [];
    return value.split(",").map((s) => s.trim()).filter(Boolean);
}

let activePopup = null;
let activeOutsideHandler = null;

function closeActivePopup() {
    if (activePopup) {
        activePopup.remove();
        activePopup = null;
    }
    if (activeOutsideHandler) {
        document.removeEventListener("pointerdown", activeOutsideHandler, true);
        activeOutsideHandler = null;
    }
}

function niceLabel(name) {
    return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function openPopup(node, widget, options) {
    closeActivePopup();

    const canvasEl = app.canvas.canvas;
    const rect = canvasEl.getBoundingClientRect();
    const ds = app.canvas.ds;
    const screenX = rect.left + (node.pos[0] + ds.offset[0]) * ds.scale;
    const screenY = rect.top + (node.pos[1] + node.size[1] + ds.offset[1]) * ds.scale;

    const popup = document.createElement("div");
    popup.className = "gr-multiselect-popup";
    Object.assign(popup.style, {
        position: "fixed",
        left: `${Math.max(4, screenX)}px`,
        top: `${screenY + 4}px`,
        width: "260px",
        maxHeight: "340px",
        overflowY: "auto",
        background: "#1a1a1a",
        border: "1px solid #555",
        borderRadius: "6px",
        padding: "6px",
        zIndex: 10000,
        boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
        fontFamily: "Arial, sans-serif",
        fontSize: "12px",
        color: "#ddd",
    });

    const title = document.createElement("div");
    title.textContent = widget.label || niceLabel(widget.name);
    Object.assign(title.style, {
        fontWeight: "bold",
        marginBottom: "6px",
        color: "#fff",
    });
    popup.appendChild(title);

    const searchBox = document.createElement("input");
    searchBox.type = "text";
    searchBox.placeholder = "Search...";
    Object.assign(searchBox.style, {
        width: "100%",
        boxSizing: "border-box",
        marginBottom: "6px",
        padding: "4px 6px",
        background: "#2a2a2a",
        border: "1px solid #444",
        borderRadius: "4px",
        color: "#eee",
    });
    popup.appendChild(searchBox);

    const btnRow = document.createElement("div");
    Object.assign(btnRow.style, { display: "flex", gap: "6px", marginBottom: "6px" });
    const makeBtn = (label, onClick) => {
        const b = document.createElement("button");
        b.textContent = label;
        Object.assign(b.style, {
            flex: "1",
            padding: "3px",
            background: "#333",
            border: "1px solid #555",
            borderRadius: "4px",
            color: "#ddd",
            cursor: "pointer",
            fontSize: "11px",
        });
        b.onclick = onClick;
        return b;
    };

    let selected = new Set(parseCsv(widget.value).filter((v) => options.includes(v)));

    function commit() {
        widget.value = Array.from(selected).join(",");
        if (widget.callback) widget.callback(widget.value);
        if (node.graph) node.graph.setDirtyCanvas(true, true);
        else if (app.canvas) app.canvas.setDirty(true, true);
    }

    btnRow.appendChild(
        makeBtn("All", () => {
            options.forEach((o) => selected.add(o));
            rebuildList(searchBox.value);
            commit();
        })
    );
    btnRow.appendChild(
        makeBtn("None", () => {
            selected.clear();
            rebuildList(searchBox.value);
            commit();
        })
    );
    popup.appendChild(btnRow);

    const listEl = document.createElement("div");
    popup.appendChild(listEl);

    function rebuildList(filter = "") {
        listEl.innerHTML = "";
        const f = filter.toLowerCase();
        for (const opt of options) {
            if (f && !opt.toLowerCase().includes(f)) continue;

            const row = document.createElement("label");
            Object.assign(row.style, {
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "3px 4px",
                cursor: "pointer",
                borderRadius: "3px",
            });
            row.onmouseenter = () => (row.style.background = "#2a2a2a");
            row.onmouseleave = () => (row.style.background = "transparent");

            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.checked = selected.has(opt);
            cb.onchange = () => {
                if (cb.checked) selected.add(opt);
                else selected.delete(opt);
                commit();
            };

            const span = document.createElement("span");
            span.textContent = opt;

            row.appendChild(cb);
            row.appendChild(span);
            listEl.appendChild(row);
        }
        if (!listEl.children.length) {
            const none = document.createElement("div");
            none.textContent = "No matches";
            none.style.opacity = "0.6";
            none.style.padding = "4px";
            listEl.appendChild(none);
        }
    }

    searchBox.oninput = () => rebuildList(searchBox.value);
    rebuildList();

    document.body.appendChild(popup);
    activePopup = popup;

    activeOutsideHandler = (e) => {
        if (activePopup && !activePopup.contains(e.target)) {
            closeActivePopup();
        }
    };
    // defer so the click that opened the popup doesn't immediately close it
    setTimeout(() => document.addEventListener("pointerdown", activeOutsideHandler, true), 0);
}

function createMultiSelectWidget(inputName, options, defaultValue, label) {
    const widget = {
        name: inputName,
        type: "GR_MULTI_SELECT",
        value: defaultValue || "",
        label: label,
        options: { serialize: true },

        draw(ctx, node, widgetWidth, y, widgetHeight) {
            const margin = 10;
            const H = widgetHeight || (LiteGraph.NODE_WIDGET_HEIGHT || 20);

            ctx.save();
            ctx.fillStyle = "#222";
            if (ctx.roundRect) {
                ctx.beginPath();
                ctx.roundRect(margin, y, widgetWidth - margin * 2, H, 4);
                ctx.fill();
            } else {
                ctx.fillRect(margin, y, widgetWidth - margin * 2, H);
            }

            const selectedCount = parseCsv(this.value).length;
            const countText =
                selectedCount === 0
                    ? "none"
                    : selectedCount === options.length
                    ? "all"
                    : `${selectedCount} selected`;

            ctx.font = "12px Arial";
            ctx.fillStyle = "#ccc";
            ctx.textAlign = "left";
            ctx.fillText(this.label || niceLabel(this.name), margin + 8, y + H * 0.65);

            ctx.fillStyle = selectedCount ? "#8fdc8f" : "#888";
            ctx.textAlign = "right";
            ctx.fillText(countText, widgetWidth - margin - 16, y + H * 0.65);

            ctx.fillStyle = "#aaa";
            ctx.beginPath();
            const ax = widgetWidth - margin - 8;
            const ay = y + H / 2;
            ctx.moveTo(ax - 4, ay - 3);
            ctx.lineTo(ax + 4, ay - 3);
            ctx.lineTo(ax, ay + 3);
            ctx.fill();
            ctx.restore();
        },

        mouse(event, pos, node) {
            if (event.type === "pointerdown" || event.type === "pointerup" || event.type === "click") {
                if (event.type === "pointerdown") {
                    openPopup(node, this, options);
                }
                return true;
            }
            return false;
        },

        computeSize(widgetWidth) {
            return [widgetWidth, LiteGraph.NODE_WIDGET_HEIGHT || 20];
        },
    };
    return widget;
}

app.registerExtension({
    name: "GraftingRayman.PromptReplacerAttributesMulti",
    async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const optionalInputs = nodeData.input?.optional || {};
            for (const [inputName, def] of Object.entries(optionalInputs)) {
                const [type, config] = def;
                if (type !== "STRING" || !config || !config.multi_select) continue;

                const idx = this.widgets ? this.widgets.findIndex((w) => w.name === inputName) : -1;
                if (idx === -1) continue;

                const options = config.options || [];
                const currentValue = this.widgets[idx].value;
                const newWidget = createMultiSelectWidget(inputName, options, currentValue, config.label);
                this.widgets[idx] = newWidget;
            }

            this.setSize(this.computeSize());
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            closeActivePopup();
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };
    },
});
