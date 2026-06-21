import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { useAsync, useRunWebSocket } from './hooks';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

describe('useAsync', () => {
  it('keeps existing data visible while a live refresh is pending', async () => {
    const first = deferred<string>();
    const second = deferred<string>();
    const loaders = [first, second];
    const { result, rerender } = renderHook(
      ({ version }) => useAsync(() => loaders[version].promise, [version]),
      { initialProps: { version: 0 } }
    );

    expect(result.current.loading).toBe(true);

    act(() => first.resolve('initial'));
    await waitFor(() => expect(result.current.data).toBe('initial'));
    expect(result.current.loading).toBe(false);

    rerender({ version: 1 });

    expect(result.current.data).toBe('initial');
    expect(result.current.loading).toBe(false);
    expect(result.current.refreshing).toBe(true);
  });
});

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  readonly url: string;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  close = vi.fn(() => {
    this.onclose?.();
  });

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }
}

describe('useRunWebSocket', () => {
  const originalWebSocket = window.WebSocket;

  afterEach(() => {
    window.WebSocket = originalWebSocket;
    MockWebSocket.instances = [];
  });

  it('connects to the run live endpoint and records the latest event', async () => {
    window.WebSocket = MockWebSocket as unknown as typeof WebSocket;

    const { result, unmount } = renderHook(() => useRunWebSocket('run_a', true));

    expect(MockWebSocket.instances).toHaveLength(1);
    expect(MockWebSocket.instances[0].url).toMatch(/\/ws\/runs\/run_a\/live$/);

    act(() => {
      MockWebSocket.instances[0].onopen?.();
    });
    expect(result.current.status).toBe('connected');

    act(() => {
      MockWebSocket.instances[0].onmessage?.(
        new MessageEvent('message', { data: JSON.stringify({ table: 'reward_terms', metric_name: 'reward_so_far' }) })
      );
    });

    await waitFor(() => expect(result.current.lastEvent?.metric_name).toBe('reward_so_far'));
    expect(result.current.eventCount).toBe(1);

    unmount();
    expect(MockWebSocket.instances[0].close).toHaveBeenCalled();
  });

  it('does not connect while live mode is frozen', () => {
    window.WebSocket = MockWebSocket as unknown as typeof WebSocket;

    const { result } = renderHook(() => useRunWebSocket('run_a', false));

    expect(MockWebSocket.instances).toHaveLength(0);
    expect(result.current.status).toBe('disabled');
  });
});
