import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (m) => m.slice(1))), "..");
const SOURCE = path.join(ROOT, "entregables", "Presentacion_Proyecto_2_MosquitoAlert.pptx");
const RESULT_DIR = path.join(ROOT, "entregables", "resultados");
const FIGURE_DIR = path.join(ROOT, "entregables", "figuras");
const SCREENSHOT_DIR = path.join(ROOT, "entregables", "capturas");
const BUILD_DIR = path.join(ROOT, ".codex-pptx-build");
const FINAL_PPTX = path.join(ROOT, "entregables", "Presentacion_Final_Proyecto_2_MosquitoAlert.pptx");
const SKILL_DIR = "C:/Users/VICTUS/.codex/plugins/cache/openai-primary-runtime/presentations/26.1009.11318/skills/presentations";
const RUNTIME_PYTHON = "C:/Users/VICTUS/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
process.env.RUNTIME_NODE_MODULES ??= path.resolve(ROOT, "..", "node_modules");

const GREEN = "#173A2B";
const MID_GREEN = "#2D5A45";
const SOFT_GREEN = "#7B9B83";
const GOLD = "#C3913C";
const RUST = "#A64B38";
const TEXT = "#23312A";
const GRAY = "#66736C";
const PALE = "#F2F5F2";
const LINE = "#D9D9D9";
const WHITE = "#FFFFFF";
const FONT = "Arial";

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') {
        value += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[i + 1] === "\n") i += 1;
      row.push(value);
      value = "";
      if (row.some((cell) => cell !== "")) rows.push(row);
      row = [];
    } else {
      value += char;
    }
  }
  if (value || row.length) {
    row.push(value);
    rows.push(row);
  }
  const [headers, ...records] = rows;
  return records.map((record) => Object.fromEntries(headers.map((header, index) => [header, record[index] ?? ""])));
}

function number(value) {
  return Number.parseFloat(value);
}

function chartNumber(value) {
  return Number(number(value).toFixed(4));
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: 22,
    typeface: FONT,
    color: TEXT,
    alignment: "left",
    ...style,
  };
  return shape;
}

function addTitle(slide, title, numberText) {
  slide.background.fill = "#FBFAF6";
  addText(slide, title, { left: 64, top: 48, width: 1100, height: 58 }, {
    fontSize: 44,
    bold: true,
    color: GREEN,
  });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 64, top: 118, width: 84, height: 4 },
    fill: GOLD,
    line: { style: "solid", fill: GOLD, width: 0 },
  });
  addText(slide, numberText, { left: 1180, top: 52, width: 42, height: 30 }, {
    fontSize: 18,
    bold: true,
    color: GOLD,
    alignment: "right",
  });
}

function addFooter(slide) {
  slide.shapes.add({
    geometry: "rect",
    position: { left: 64, top: 681, width: 1152, height: 1 },
    fill: LINE,
    line: { style: "solid", fill: LINE, width: 0 },
  });
  addText(slide, "Proyecto 2 · Data Science · Universidad del Valle de Guatemala", {
    left: 64, top: 688, width: 760, height: 18,
  }, { fontSize: 13, color: GRAY });
}

function addSectionLabel(slide, text, position) {
  return addText(slide, text, position, { fontSize: 18, bold: true, color: GOLD });
}

async function addImageIfPresent(slide, imagePath, alt, position) {
  try {
    await fs.access(imagePath);
    const bytes = await fs.readFile(imagePath);
    const blob = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    slide.images.add({ blob, contentType: "image/png", alt, fit: "contain", position });
    return true;
  } catch {
    return false;
  }
}

async function main() {
  await fs.mkdir(BUILD_DIR, { recursive: true });
  const presentation = await PresentationFile.importPptx(await FileBlob.load(SOURCE));
  const metrics = parseCsv(await fs.readFile(path.join(RESULT_DIR, "metricas_modelos.csv"), "utf8"));
  const perClass = parseCsv(await fs.readFile(path.join(RESULT_DIR, "metricas_por_clase.csv"), "utf8"));
  const tuning = parseCsv(await fs.readFile(path.join(RESULT_DIR, "ajuste_hiperparametros.csv"), "utf8"));
  const metadata = JSON.parse(await fs.readFile(path.join(RESULT_DIR, "metadatos_ejecucion.json"), "utf8"));
  const best = [...metrics].sort((a, b) => number(b.f1_macro) - number(a.f1_macro))[0];

  // La presentación original conserva las cinco láminas de EDA, pero la portada
  // se actualiza para que el archivo funcione como presentación final.
  const titleHit = await presentation.inspect({
    select: "$.slides[0].elements[?(@.text == 'Análisis exploratorio\\nAI Mosquito Alert Challenge 2023')]",
    include: ["id", "text"],
  });
  if (titleHit.records[0]?.id) {
    presentation.resolve(titleHit.records[0].id).text = "Resultados y análisis\nAI Mosquito Alert Challenge 2023";
  }
  const edaHit = await presentation.inspect({
    select: "$.slides[0].elements[?(@.text == 'EDA')]",
    include: ["id", "text"],
  });
  if (edaHit.records[0]?.id) presentation.resolve(edaHit.records[0].id).text = "ML";
  const dateHit = await presentation.inspect({
    select: "$.slides[0].elements[?(@.text == 'Guatemala · Septiembre 2026')]",
    include: ["id", "text"],
  });
  if (dateHit.records[0]?.id) presentation.resolve(dateHit.records[0].id).text = "Guatemala · Octubre 2026";
  const lastEdaTitle = await presentation.inspect({
    select: "$.slides[5].elements[?(@.text == 'Conclusiones y siguiente etapa')]",
    include: ["id", "text"],
  });
  if (lastEdaTitle.records[0]?.id) {
    presentation.resolve(lastEdaTitle.records[0].id).text = "Conclusiones del análisis exploratorio";
  }

  // 07 Metodología
  {
    const slide = presentation.slides.add();
    addTitle(slide, "Metodología de modelado", "07");
    const phases = [
      ["01", "Entrada", "Corrección EXIF y recorte del mosquito con 5% de margen."],
      ["02", "Representación", "Letterbox de 96 × 96, HOG para forma e HSV para color."],
      ["03", "Particiones", `${Number(metadata.registros_totales).toLocaleString("es-GT")} imágenes; división estratificada y máximo de 600 por clase en entrenamiento.`],
      ["04", "Selección", "Dos configuraciones por familia; criterio principal: F1 macro de validación."],
    ];
    for (let i = 0; i < phases.length; i += 1) {
      const top = 164 + i * 116;
      addText(slide, phases[i][0], { left: 64, top, width: 66, height: 40 }, { fontSize: 28, bold: true, color: GOLD });
      addText(slide, phases[i][1], { left: 148, top, width: 250, height: 36 }, { fontSize: 26, bold: true, color: GREEN });
      addText(slide, phases[i][2], { left: 148, top: top + 38, width: 475, height: 56 }, { fontSize: 19, color: GRAY });
    }
    slide.shapes.add({
      geometry: "rect",
      position: { left: 680, top: 162, width: 500, height: 430 },
      fill: GREEN,
      line: { style: "solid", fill: GREEN, width: 0 },
    });
    addSectionLabel(slide, "PARTICIONES", { left: 724, top: 205, width: 210, height: 28 });
    addText(slide, `${metadata.particiones.train.toLocaleString("es-GT")}`, { left: 724, top: 252, width: 180, height: 72 }, { fontSize: 55, bold: true, color: WHITE });
    addText(slide, "entrenamiento", { left: 724, top: 322, width: 200, height: 34 }, { fontSize: 21, color: "#F0E4CF" });
    addText(slide, `${metadata.particiones.validation.toLocaleString("es-GT")} validación`, { left: 724, top: 392, width: 300, height: 38 }, { fontSize: 27, bold: true, color: WHITE });
    addText(slide, `${metadata.particiones.test.toLocaleString("es-GT")} prueba`, { left: 724, top: 448, width: 300, height: 38 }, { fontSize: 27, bold: true, color: WHITE });
    addText(slide, `${metadata.cajas_ajustadas_a_limites} cajas fuera de límites se ajustan al recortar`, { left: 724, top: 522, width: 400, height: 44 }, { fontSize: 18, color: "#F0E4CF" });
    addFooter(slide);
    slide.speakerNotes.text = [
      "La división evita que los parámetros se escojan con la misma partición usada para informar resultados.",
      "El muestreo limita únicamente entrenamiento; validación y prueba conservan el desbalance original.",
      "Fuente: scripts src/modeling.py y src/train_models.py del proyecto.",
    ];
  }

  // 08 Ajuste de hiperparámetros
  {
    const slide = presentation.slides.add();
    addTitle(slide, "Ajuste de hiperparámetros", "08");
    const sorted = [...tuning].sort((a, b) => a.modelo.localeCompare(b.modelo) || number(b.f1_macro_validacion) - number(a.f1_macro_validacion));
    const values = [["Modelo", "Configuración", "Rasgos", "F1 macro validación"], ...sorted.map((row) => [
      row.modelo,
      row.configuracion,
      row.caracteristicas === "combined" ? "HOG + HSV" : "HSV",
      number(row.f1_macro_validacion).toFixed(3),
    ])];
    const table = slide.tables.add({
      rows: values.length,
      columns: 4,
      left: 64,
      top: 164,
      width: 1152,
      height: 365,
      columnWidths: [260, 355, 220, 317],
      values,
    });
    table.borders.assign({ style: "solid", fill: LINE, width: 1 });
    table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 4 }).assign({
      fill: GREEN,
      textStyle: { typeface: FONT, fontSize: 16, bold: true, color: WHITE },
      margins: { left: 8, right: 8, top: 7, bottom: 7 },
      anchor: "middle",
    });
    for (let row = 1; row < values.length; row += 1) {
      table.cells.block({ row, column: 0, rowCount: 1, columnCount: 4 }).assign({
        fill: row % 2 ? WHITE : PALE,
        textStyle: { typeface: FONT, fontSize: 15, color: TEXT },
        margins: { left: 8, right: 8, top: 6, bottom: 6 },
        anchor: "middle",
      });
    }
    addText(slide, "Selección", { left: 64, top: 556, width: 180, height: 30 }, { fontSize: 19, bold: true, color: GOLD });
    addText(slide, "Se elige la mejor configuración de cada familia por F1 macro en validación; después se reajusta con entrenamiento más validación y se evalúa una vez en prueba.", { left: 64, top: 592, width: 1110, height: 64 }, { fontSize: 20, color: TEXT });
    addFooter(slide);
    slide.speakerNotes.text = [
      "El ajuste es deliberadamente acotado porque esta entrega presenta baselines parciales ejecutables en CPU.",
      "Fuente: entregables/resultados/ajuste_hiperparametros.csv.",
    ];
  }

  // 09 Comparación de métricas
  {
    const slide = presentation.slides.add();
    addTitle(slide, "Comparación de modelos", "09");
    const categories = metrics.map((row) => row.modelo);
    const chart = slide.charts.add("bar", {
      position: { left: 52, top: 158, width: 800, height: 446 },
      categories,
      series: [
        { name: "F1 macro", values: metrics.map((row) => chartNumber(row.f1_macro)), fill: GREEN },
        { name: "Exactitud balanceada", values: metrics.map((row) => chartNumber(row.exactitud_balanceada)), fill: GOLD },
        { name: "Exactitud global", values: metrics.map((row) => chartNumber(row.exactitud)), fill: SOFT_GREEN },
      ],
      barOptions: { direction: "bar", grouping: "clustered", gapWidth: 45 },
      hasLegend: true,
      legend: { position: "bottom", overlay: false, textStyle: { fontSize: 13, fill: GRAY } },
      xAxis: { min: 0, max: 1, majorUnit: 0.2, numberFormatCode: "0.0", majorGridlines: { style: "solid", fill: LINE, width: 1 }, textStyle: { fontSize: 13, fill: GRAY } },
      yAxis: { textStyle: { fontSize: 14, fill: TEXT }, line: { style: "solid", fill: LINE, width: 1 } },
      dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.00", textStyle: { fontSize: 12, fill: TEXT } },
      chartFill: "#FBFAF6",
      plotAreaFill: "#FBFAF6",
      chartLine: { style: "solid", fill: "none", width: 0 },
    });
    const utils = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href);
    utils.applyPresentationChartFont(chart, { fontFamily: FONT });
    addSectionLabel(slide, "MEJOR BASELINE", { left: 900, top: 178, width: 240, height: 28 });
    addText(slide, best.modelo, { left: 884, top: 224, width: 300, height: 78 }, { fontSize: 31, bold: true, color: GREEN, alignment: "center" });
    addText(slide, number(best.f1_macro).toFixed(3), { left: 884, top: 326, width: 300, height: 74 }, { fontSize: 58, bold: true, color: GOLD, alignment: "center" });
    addText(slide, "F1 macro en prueba", { left: 904, top: 398, width: 260, height: 34 }, { fontSize: 20, color: GRAY, alignment: "center" });
    addText(slide, "La exactitud global puede favorecer las dos clases mayoritarias. La selección se apoya en promedios macro.", { left: 882, top: 486, width: 304, height: 100 }, { fontSize: 19, bold: true, color: RUST, alignment: "center" });
    addFooter(slide);
    slide.speakerNotes.text = [
      "F1 macro da el mismo peso a cada clase y evita que Albopictus y Culex dominen la comparación.",
      "Fuente: entregables/resultados/metricas_modelos.csv.",
    ];
  }

  // 10 Recall por clase
  {
    const slide = presentation.slides.add();
    addTitle(slide, "Rendimiento por clase", "10");
    const classes = [...new Set(perClass.map((row) => row.clase))];
    const modelNames = metrics.map((row) => row.modelo);
    const chart = slide.charts.add("bar", {
      position: { left: 52, top: 160, width: 820, height: 450 },
      categories: classes,
      series: modelNames.map((modelName, index) => ({
        name: modelName,
        values: classes.map((className) => chartNumber(perClass.find((row) => row.modelo === modelName && row.clase === className)?.recall ?? 0)),
        fill: [GREEN, GOLD, SOFT_GREEN][index],
      })),
      barOptions: { direction: "column", grouping: "clustered", gapWidth: 55 },
      hasLegend: true,
      legend: { position: "bottom", overlay: false, textStyle: { fontSize: 12, fill: GRAY } },
      xAxis: { textStyle: { fontSize: 11, fill: TEXT }, line: { style: "solid", fill: LINE, width: 1 } },
      yAxis: { min: 0, max: 1, majorUnit: 0.2, numberFormatCode: "0.0", majorGridlines: { style: "solid", fill: LINE, width: 1 }, textStyle: { fontSize: 12, fill: GRAY } },
      chartFill: "#FBFAF6",
      plotAreaFill: "#FBFAF6",
      chartLine: { style: "solid", fill: "none", width: 0 },
    });
    const utils = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href);
    utils.applyPresentationChartFont(chart, { fontFamily: FONT });
    const bestRows = perClass.filter((row) => row.modelo === best.modelo).sort((a, b) => number(a.recall) - number(b.recall));
    const weakest = bestRows[0];
    addSectionLabel(slide, "LECTURA PRINCIPAL", { left: 920, top: 178, width: 230, height: 28 });
    addText(slide, weakest.clase, { left: 900, top: 238, width: 280, height: 54 }, { fontSize: 29, bold: true, color: GREEN, alignment: "center" });
    addText(slide, number(weakest.recall).toFixed(2), { left: 900, top: 314, width: 280, height: 72 }, { fontSize: 56, bold: true, color: RUST, alignment: "center" });
    addText(slide, `recall más bajo del modelo seleccionado\nsoporte de prueba: ${weakest.soporte}`, { left: 900, top: 394, width: 280, height: 78 }, { fontSize: 19, color: GRAY, alignment: "center" });
    addText(slide, "Los pocos ejemplos de Aegypti y Anopheles vuelven inestables sus métricas. Cada error pesa mucho más.", { left: 900, top: 516, width: 280, height: 84 }, { fontSize: 19, bold: true, color: TEXT, alignment: "center" });
    addFooter(slide);
    slide.speakerNotes.text = [
      "El recall responde qué proporción de cada clase real logra recuperar el modelo.",
      "Fuente: entregables/resultados/metricas_por_clase.csv.",
    ];
  }

  // 11 Confusiones
  {
    const slide = presentation.slides.add();
    addTitle(slide, "Dónde se equivocan los modelos", "11");
    await addImageIfPresent(
      slide,
      path.join(FIGURE_DIR, "03_matrices_confusion.png"),
      "Matrices de confusión normalizadas de tres modelos",
      { left: 60, top: 152, width: 1160, height: 462 },
    );
    addText(slide, "Cada fila suma 1. Los valores de la diagonal representan aciertos por clase; las celdas fuera de ella muestran hacia qué grupo se desplaza el error.", { left: 114, top: 620, width: 1050, height: 42 }, { fontSize: 18, color: GRAY, alignment: "center" });
    addFooter(slide);
    slide.speakerNotes.text = [
      "La normalización por fila evita que el tamaño de las clases mayoritarias oculte los errores de las minoritarias.",
      "Fuente: entregables/figuras/03_matrices_confusion.png.",
    ];
  }

  // 12 Aplicación
  {
    const slide = presentation.slides.add();
    addTitle(slide, "Aplicación interactiva", "12");
    const leftOk = await addImageIfPresent(
      slide,
      path.join(SCREENSHOT_DIR, "app_clasificacion.png"),
      "Pantalla de clasificación de la aplicación Streamlit",
      { left: 64, top: 164, width: 552, height: 390 },
    );
    const rightOk = await addImageIfPresent(
      slide,
      path.join(SCREENSHOT_DIR, "app_rendimiento.png"),
      "Pantalla de métricas interactivas de la aplicación Streamlit",
      { left: 664, top: 164, width: 552, height: 390 },
    );
    if (!leftOk) addText(slide, "Captura de clasificación pendiente", { left: 64, top: 310, width: 552, height: 50 }, { fontSize: 24, color: GRAY, alignment: "center" });
    if (!rightOk) addText(slide, "Captura de rendimiento pendiente", { left: 664, top: 310, width: 552, height: 50 }, { fontSize: 24, color: GRAY, alignment: "center" });
    addText(slide, "Clasificar", { left: 64, top: 574, width: 552, height: 32 }, { fontSize: 22, bold: true, color: GREEN, alignment: "center" });
    addText(slide, "Comparar modelos", { left: 664, top: 574, width: 552, height: 32 }, { fontSize: 22, bold: true, color: GREEN, alignment: "center" });
    addText(slide, "La aplicación también permite explorar variables limpias y ocultar las métricas cuando el usuario no desea verlas.", { left: 160, top: 624, width: 960, height: 36 }, { fontSize: 18, color: GRAY, alignment: "center" });
    addFooter(slide);
    slide.speakerNotes.text = [
      "La entrada se recorta opcionalmente y pasa por el mismo preprocesamiento del entrenamiento.",
      "La salida es experimental y no sustituye una identificación entomológica experta.",
      "Fuente: app.py del proyecto.",
    ];
  }

  // 13 Conclusiones
  {
    const slide = presentation.slides.add();
    addTitle(slide, "Conclusiones y siguiente iteración", "13");
    slide.shapes.add({
      geometry: "rect",
      position: { left: 64, top: 160, width: 528, height: 450 },
      fill: GREEN,
      line: { style: "solid", fill: GREEN, width: 0 },
    });
    addSectionLabel(slide, "RESULTADO PARCIAL", { left: 104, top: 202, width: 270, height: 30 });
    addText(slide, best.modelo, { left: 104, top: 252, width: 418, height: 82 }, { fontSize: 33, bold: true, color: WHITE });
    addText(slide, `${number(best.f1_macro).toFixed(3)} F1 macro`, { left: 104, top: 358, width: 418, height: 58 }, { fontSize: 39, bold: true, color: "#F0E4CF" });
    addText(slide, `Exactitud balanceada ${number(best.exactitud_balanceada).toFixed(3)}`, { left: 104, top: 430, width: 418, height: 44 }, { fontSize: 24, color: WHITE });
    addText(slide, "El baseline establece una referencia reproducible, pero las clases minoritarias todavía requieren más datos y una evaluación repetida.", { left: 104, top: 510, width: 418, height: 76 }, { fontSize: 20, color: "#F0E4CF" });
    addText(slide, "Siguiente iteración", { left: 664, top: 174, width: 470, height: 40 }, { fontSize: 28, bold: true, color: GREEN });
    addText(slide,
      "Entrenar una red ligera con transferencia de aprendizaje.\n\nAplicar aumentación y pérdidas sensibles al desbalance.\n\nCalibrar probabilidades y repetir la validación.\n\nComparar el recorte anotado contra la fotografía completa.\n\nMantener revisión humana para predicciones de baja confianza.",
      { left: 666, top: 238, width: 500, height: 330 },
      { fontSize: 22, color: TEXT },
    );
    addText(slide, "La aplicación ya permite probar los tres modelos y auditar sus errores.", { left: 664, top: 588, width: 510, height: 42 }, { fontSize: 19, bold: true, color: RUST });
    addFooter(slide);
    slide.speakerNotes.text = [
      "Los tres modelos completan la fase de resultados parciales y dejan un benchmark para una CNN posterior.",
      "No se debe presentar el porcentaje global como si representara por igual a las seis clases.",
    ];
  }

  const renderDir = path.join(BUILD_DIR, "rendered");
  await fs.mkdir(renderDir, { recursive: true });
  for (const [index, slide] of presentation.slides.items.entries()) {
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    const file = path.join(renderDir, `slide-${String(index + 1).padStart(2, "0")}.png`);
    await fs.writeFile(file, new Uint8Array(await png.arrayBuffer()));
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 0.35 });
  await fs.writeFile(path.join(renderDir, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));

  const candidatePath = path.join(BUILD_DIR, "candidate.pptx");
  await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
  const sourceSha256 = crypto.createHash("sha256").update(await fs.readFile(SOURCE)).digest("hex");
  const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href);
  const result = await finalizePresentation({
    workspaceDir: ROOT,
    candidatePath,
    finalPath: FINAL_PPTX,
    pythonExecutable: RUNTIME_PYTHON,
    integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
    layoutArgs: [
      "--expected-slide-size-emu", "12192000,6858000",
      "--validate-bullet-geometry",
      "--validate-heading-fit",
      "--require-native-table-slide", "8",
    ],
    explicitTotalSlideCount: 13,
    requiredNativeTableOwnerSlides: [8],
    requiredNativeChartOwnerSlides: [9, 10],
    sourceTemplatePath: SOURCE,
    requiredTemplateReferenceSlides: [1, 2, 3, 4, 5, 6],
    minimumTemplateCoverageRatio: 6 / 13,
    requireExactTemplateDimensions: true,
    materializeLiteralChartWorkbooks: true,
    fontPolicy: {
      basis: "reference",
      families: [FONT],
      referencePath: SOURCE,
      referenceSha256: sourceSha256,
    },
    verifyArtifactToolImport: true,
    receiptPath: path.join(BUILD_DIR, "validation_final.json"),
  });
  console.log(JSON.stringify({ final: FINAL_PPTX, validation: result }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
