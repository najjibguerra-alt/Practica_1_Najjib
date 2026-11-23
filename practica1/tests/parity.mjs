import fs from "node:fs";
import { createModel, predict } from "../pwa/inference.js";

const model = createModel(JSON.parse(fs.readFileSync(new URL("../pwa/model.json", import.meta.url), "utf8")));
const cases = JSON.parse(fs.readFileSync(new URL("./parity_cases.json", import.meta.url), "utf8"));
const out = {};
for (const item of cases) {
  const result = predict(model, item.text);
  out[item.id] = {
    probability: result.probability,
    label: result.label,
    rejectReason: result.rejectReason,
    tokenCount: result.tokenCount,
    featureCount: result.featureCount,
  };
}
process.stdout.write(JSON.stringify(out));
