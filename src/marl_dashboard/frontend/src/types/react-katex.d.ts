declare module 'react-katex' {
  import type { ComponentType, ReactNode } from 'react';

  export type MathComponentProps = {
    math?: string;
    children?: string;
    errorColor?: string;
    renderError?: (error: Error) => ReactNode;
  };

  export const BlockMath: ComponentType<MathComponentProps>;
  export const InlineMath: ComponentType<MathComponentProps>;
}
