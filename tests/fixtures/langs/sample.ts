export interface Shape { area(): number; }

export class Widget implements Shape {
  area(): number { return 1; }
}

export function computeTotal(items: number[]): number {
  return items.reduce((a, b) => a + b, 0);
}
