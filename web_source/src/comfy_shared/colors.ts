/**
 * Color utilities
 */

type RGB = [number, number, number]

function getBrightness(rgb: RGB): number {
  return Math.round(
    (Number.parseInt(String(rgb[0])) * 299 +
      Number.parseInt(String(rgb[1])) * 587 +
      Number.parseInt(String(rgb[2])) * 114) /
      1000,
  )
}

export function isColorBright(rgb: RGB, threshold = 240): boolean {
  const brightness = getBrightness(rgb)
  return brightness > threshold
}
