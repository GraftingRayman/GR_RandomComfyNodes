import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "GraftingRayman.GRLoadAudioWithDuration.uploadButton",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GRLoadAudioWithDuration") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const audioWidget = this.widgets.find((w) => w.name === "audio");
            if (!audioWidget) return;

            const fileInput = document.createElement("input");
            Object.assign(fileInput, {
                type: "file",
                accept: "audio/*,.wav,.mp3,.ogg,.flac,.m4a,.aiff,.aif",
                style: "display: none",
                onchange: async () => {
                    if (!fileInput.files.length) return;
                    const file = fileInput.files[0];

                    const formData = new FormData();
                    formData.append("image", file);
                    formData.append("type", "input");
                    formData.append("subfolder", "");

                    try {
                        const resp = await api.fetchApi("/upload/image", {
                            method: "POST",
                            body: formData,
                        });

                        if (resp.status !== 200) {
                            const err = await resp.text();
                            alert(`Upload failed: ${err}`);
                            return;
                        }

                        const data = await resp.json();
                        const fullName = data.subfolder
                            ? `${data.subfolder}/${data.name}`
                            : data.name;

                        if (!audioWidget.options.values.includes(fullName)) {
                            audioWidget.options.values.push(fullName);
                        }
                        audioWidget.value = fullName;
                        if (audioWidget.callback) {
                            audioWidget.callback(fullName);
                        }
                        this.graph?.setDirtyCanvas(true, true);
                    } catch (e) {
                        alert(`Upload error: ${e}`);
                    }
                },
            });
            document.body.append(fileInput);

            const uploadWidget = this.addWidget(
                "button",
                "choose audio to upload",
                "",
                () => {
                    fileInput.click();
                }
            );
            uploadWidget.serialize = false;

            this.onRemoved = function () {
                fileInput.remove();
            };
        };
    },
});
