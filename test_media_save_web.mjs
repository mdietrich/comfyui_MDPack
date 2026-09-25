// Tests for web/mediaSaveMigration.js.
// Run with: node test_media_save_web.mjs

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const failures = [];

function check(name, condition, detail = "") {
    if (condition) {
        console.log(`PASS ${name}`);
    } else {
        failures.push(name);
        console.log(`FAIL ${name} ${detail}`);
    }
}

const testDirectory = path.dirname(fileURLToPath(import.meta.url));
const sourcePath = path.join(testDirectory, "web", "mediaSaveMigration.js");
const sourceText = fs.readFileSync(sourcePath, "utf-8");
const scratchDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "mdpack-media-web-"));
const stubAppPath = path.join(scratchDirectory, "stub-app.js");
fs.writeFileSync(
    stubAppPath,
    `export const app = {
        registerExtension(extension) {
            app.registeredExtension = extension;
        },
    };`,
    "utf-8",
);
const rewrittenPath = path.join(scratchDirectory, "mediaSaveMigration.mjs");
fs.writeFileSync(
    rewrittenPath,
    sourceText.replace(
        'import { app } from "../../scripts/app.js";',
        `import { app } from ${JSON.stringify(pathToFileURL(stubAppPath).href)};`,
    ),
    "utf-8",
);

const migrationModule = await import(pathToFileURL(rewrittenPath).href);
const {
    IMAGE_NODE_NAME,
    VIDEO_NODE_NAME,
    OLD_IMAGE_WIDGET_NAMES,
    OLD_VIDEO_WIDGET_NAMES,
    removeLegacyApiWidgets,
    migrateMediaSaveWidgetValues,
} = migrationModule;

function nodeWithWidgets(widgetNames) {
    return {
        widgets: widgetNames.map((name) => ({ name, value: `default-${name}` })),
    };
}

function valuesOf(node) {
    return Object.fromEntries(node.widgets.map((widget) => [widget.name, widget.value]));
}

const imageCurrentNames = OLD_IMAGE_WIDGET_NAMES.filter(
    (name) => name !== "c2patool_path" && name !== "tool_timeout_seconds",
);
const oldImageValues = [
    "legacy/image",
    "jpeg",
    91,
    true,
    "Fully AI-generated",
    false,
    "certificate.pem",
    "private.key",
    "es256",
    "/obsolete/c2patool",
    17,
    false,
];
const imageNode = nodeWithWidgets(imageCurrentNames);
check(
    "old image graph is migrated",
    migrateMediaSaveWidgetValues(
        imageNode,
        IMAGE_NODE_NAME,
        { widgets_values: oldImageValues },
    ) === true,
);
const migratedImageValues = valuesOf(imageNode);
check(
    "image migration preserves workflow choice without shifting legacy tool values",
    migratedImageValues.embed_workflow === false &&
        migratedImageValues.signing_algorithm === "es256" &&
        !Object.values(migratedImageValues).includes("/obsolete/c2patool"),
    JSON.stringify(migratedImageValues),
);
const imageValuesAfterFirstMigration = JSON.stringify(migratedImageValues);
migrateMediaSaveWidgetValues(
    imageNode,
    IMAGE_NODE_NAME,
    { widgets_values: oldImageValues },
);
check(
    "loading the same old image graph repeatedly is idempotent",
    JSON.stringify(valuesOf(imageNode)) === imageValuesAfterFirstMigration,
);

const videoCurrentNames = OLD_VIDEO_WIDGET_NAMES.filter(
    (name) => name !== "c2patool_path",
);
const oldVideoValues = [
    "legacy/video",
    24,
    true,
    "AI-edited / composite",
    true,
    "certificate.pem",
    "private.key",
    "es256",
    "c2patool",
    47,
    true,
    "ffmpeg-custom",
    "ffprobe-custom",
    "exiftool-custom",
    "remux_all_streams",
];
const videoNode = nodeWithWidgets(videoCurrentNames);
check(
    "old video graph is migrated",
    migrateMediaSaveWidgetValues(
        videoNode,
        VIDEO_NODE_NAME,
        { widgets_values: oldVideoValues },
    ) === true,
);
const migratedVideoValues = valuesOf(videoNode);
check(
    "video migration removes only c2patool and preserves timeout plus workflow choice",
    migratedVideoValues.tool_timeout_seconds === 47 &&
        migratedVideoValues.embed_workflow === true &&
        migratedVideoValues.ffmpeg_path === "ffmpeg-custom" &&
        !Object.values(migratedVideoValues).includes("c2patool"),
    JSON.stringify(migratedVideoValues),
);

const currentImageValues = oldImageValues.filter((value, index) => index !== 9 && index !== 10);
const currentImageNode = nodeWithWidgets(imageCurrentNames);
const currentDefaults = JSON.stringify(valuesOf(currentImageNode));
check(
    "current image layout is not remigrated",
    migrateMediaSaveWidgetValues(
        currentImageNode,
        IMAGE_NODE_NAME,
        { widgets_values: currentImageValues },
    ) === false && JSON.stringify(valuesOf(currentImageNode)) === currentDefaults,
);
check(
    "unknown future layout is left untouched",
    migrateMediaSaveWidgetValues(
        currentImageNode,
        IMAGE_NODE_NAME,
        { widgets_values: [...currentImageValues, "future-value"] },
    ) === false,
);

const aliasesNode = nodeWithWidgets([
    ...imageCurrentNames,
    "c2patool_path",
    "tool_timeout_seconds",
]);
check(
    "legacy image API aliases are removed from the graph UI",
    removeLegacyApiWidgets(aliasesNode, IMAGE_NODE_NAME) === true &&
        !aliasesNode.widgets.some((widget) =>
            widget.name === "c2patool_path" || widget.name === "tool_timeout_seconds"),
);
check(
    "removing already absent legacy widgets is harmless",
    removeLegacyApiWidgets(aliasesNode, IMAGE_NODE_NAME) === false,
);

console.log();
if (failures.length) {
    console.log(`${failures.length} TEST(S) FAILED: ${failures.join(", ")}`);
    process.exit(1);
}
console.log("ALL TESTS PASSED");
