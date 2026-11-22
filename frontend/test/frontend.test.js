import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const distDir = path.resolve(__dirname, '../dist');

test('Frontend production build bundle exists and is valid', () => {
  const indexHtmlPath = path.join(distDir, 'index.html');
  assert.ok(fs.existsSync(indexHtmlPath), 'dist/index.html must exist');

  const html = fs.readFileSync(indexHtmlPath, 'utf-8');
  assert.ok(html.includes('<div id="root"></div>'), 'index.html must contain root mount div');
  assert.ok(html.includes('assets/index-'), 'index.html must reference bundled JS/CSS assets');

  const assetsDir = path.join(distDir, 'assets');
  assert.ok(fs.existsSync(assetsDir), 'dist/assets directory must exist');

  const files = fs.readdirSync(assetsDir);
  const jsBundle = files.find(f => f.endsWith('.js'));
  const cssBundle = files.find(f => f.endsWith('.css'));

  assert.ok(jsBundle, 'JavaScript bundle must be present');
  assert.ok(cssBundle, 'CSS bundle must be present');

  const jsContent = fs.readFileSync(path.join(assetsDir, jsBundle), 'utf-8');
  assert.ok(jsContent.includes('/api/v1/score'), 'Bundle must reference scoring API endpoint');
  assert.ok(jsContent.includes('/api/v1/model-info'), 'Bundle must reference model-info API endpoint');
  assert.ok(jsContent.includes('/api/v1/demo/samples'), 'Bundle must reference demo samples API endpoint');
});
