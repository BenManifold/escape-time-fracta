/**
 * LAN HTTPS static server for phones (WebGPU needs a secure context).
 * Usage: npm install && npm run dev:https
 * Regenerate cert after your Wi‑Fi IP changes: npm run dev:https:regen
 */
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import selfsigned from "selfsigned";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const certDir = path.join(root, "dev-certs");
const certPath = path.join(certDir, "cert.pem");
const keyPath = path.join(certDir, "key.pem");
const certFormatVersionPath = path.join(certDir, ".cert-format-version");
/** Bump when TLS extensions change so existing dev-certs/ are regenerated. */
const CERT_FORMAT_VERSION = "2";
const port = Number(process.env.DEV_HTTPS_PORT || "3443") || 3443;

const require = createRequire(import.meta.url);

function lanIPv4s() {
  const nets = os.networkInterfaces();
  const out = [];
  for (const addrs of Object.values(nets)) {
    for (const a of addrs || []) {
      if (a.family === "IPv4" && !a.internal) out.push(a.address);
    }
  }
  return [...new Set(out)];
}

function readCertFormatVersion() {
  try {
    return fs.readFileSync(certFormatVersionPath, "utf8").trim();
  } catch {
    return "";
  }
}

function ensureCerts(regen) {
  if (regen) {
    try {
      fs.unlinkSync(certPath);
    } catch {
      /* ignore */
    }
    try {
      fs.unlinkSync(keyPath);
    } catch {
      /* ignore */
    }
    try {
      fs.unlinkSync(certFormatVersionPath);
    } catch {
      /* ignore */
    }
  }

  fs.mkdirSync(certDir, { recursive: true });

  const staleFormat = readCertFormatVersion() !== CERT_FORMAT_VERSION;
  const needWrite =
    staleFormat || !fs.existsSync(certPath) || !fs.existsSync(keyPath);
  if (!needWrite) return;

  const ips = ["127.0.0.1", ...lanIPv4s()];
  const altNames = [
    { type: 2, value: "localhost" },
    ...ips.map((ip) => ({ type: 7, ip })),
  ];

  // Passing only subjectAltName replaced selfsigned defaults and produced a cert
  // without proper server EKU / keyUsage; iOS then fails TLS ("secure connection failed").
  const extensions = [
    { name: "basicConstraints", cA: false },
    {
      name: "keyUsage",
      digitalSignature: true,
      keyEncipherment: true,
    },
    { name: "extKeyUsage", serverAuth: true },
    { name: "subjectAltName", altNames },
  ];

  const p = selfsigned.generate([{ name: "commonName", value: "escape-time-fracta-dev" }], {
    keySize: 2048,
    days: 825,
    algorithm: "sha256",
    extensions,
  });

  fs.writeFileSync(keyPath, p.private, "utf8");
  fs.writeFileSync(certPath, p.cert, "utf8");
  fs.writeFileSync(certFormatVersionPath, `${CERT_FORMAT_VERSION}\n`, "utf8");

  console.log("Wrote TLS files to dev-certs/ (gitignored).");
  if (staleFormat && !regen) {
    console.log("(Regenerated: dev TLS certificate format was updated.)");
  }
}

function main() {
  const regen = process.argv.includes("--regen");
  ensureCerts(regen);

  const ips = lanIPv4s();
  console.log("");
  console.log(`HTTPS (bind all interfaces): https://0.0.0.0:${port}`);
  for (const ip of ips) {
    console.log(`  On your phone:  https://${ip}:${port}`);
  }
  if (ips.length === 0) {
    console.log("  (No LAN IPv4 found; use https://127.0.0.1 on this machine.)");
  }
  console.log("");
  console.log(
    "Trust this cert on the phone (once): AirDrop/email dev-certs/cert.pem, install the profile,",
  );
  console.log(
    "then iOS: Settings → General → About → Certificate Trust Settings → enable full trust for the cert.",
  );
  console.log("");

  let serveMain;
  try {
    serveMain = require.resolve("serve/build/main.js");
  } catch {
    console.error('Run "npm install" in the repo root first.');
    process.exit(1);
  }

  const child = spawn(
    process.execPath,
    [
      serveMain,
      ".",
      "-l",
      `tcp://0.0.0.0:${port}`,
      "--ssl-cert",
      certPath,
      "--ssl-key",
      keyPath,
    ],
    { cwd: root, stdio: "inherit" },
  );
  child.on("exit", (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    else process.exit(code ?? 0);
  });
}

main();
