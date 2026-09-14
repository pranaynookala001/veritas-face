import nextTs from "eslint-config-next/typescript";
import nextVitals from "eslint-config-next/core-web-vitals";

const eslintConfig = [
  ...nextVitals,
  ...nextTs,
  {
    ignores: [
    ".next/**",
    "build/**",
    "next-env.d.ts",
    "out/**",
    ],
  },
];

export default eslintConfig;
