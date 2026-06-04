<script setup>
// VLA viewer (edge robotics inference). Purpose-built phase dashboard.
// Stub for phase 6a; filled in by 6c (VlaConfig + VlaReadout + endpoints).
import { RouterLink } from 'vue-router'
</script>

<template>
  <div class="vla_placeholder">
    <h1>VLA Viewer</h1>
    <p>Edge-device roofline analysis for Vision-Language-Action models.</p>
    <p>Coming up: per-phase latency breakdown, control-budget &amp; memory-fit verdicts.</p>
    <RouterLink to="/">&larr; LLM viewer</RouterLink>
  </div>
</template>

<style scoped>
.vla_placeholder {
  padding: 40px;
  font-family: sans-serif;
}
</style>
