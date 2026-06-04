import { createRouter, createWebHashHistory } from 'vue-router'
import LlmView from './views/LlmView.vue'
import VlaView from './views/VlaView.vue'

// Two separate products sharing one app shell:
//   /      -> LLM viewer (cloud LLM serving) -- behavior unchanged
//   /vla   -> VLA viewer (edge robotics inference)
// Hash history keeps deep links working on plain static hosting.
const routes = [
  { path: '/', name: 'llm', component: LlmView },
  { path: '/vla', name: 'vla', component: VlaView },
]

export const router = createRouter({
  history: createWebHashHistory(),
  routes,
})
