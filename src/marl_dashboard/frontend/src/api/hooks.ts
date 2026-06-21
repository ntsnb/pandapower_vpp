import { useEffect, useRef, useState } from 'react';

import type { LiveEvent, WebSocketStatus } from './types';

export function useAsync<T>(
  loader: () => Promise<T>,
  deps: readonly unknown[]
): { data: T | null; error: string | null; loading: boolean; refreshing: boolean } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const hasData = useRef(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(!hasData.current);
    setRefreshing(hasData.current);
    loader()
      .then((value) => {
        if (!cancelled) {
          setData(value);
          hasData.current = true;
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
          setRefreshing(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, deps);

  return { data, error, loading, refreshing };
}

export function useLiveTick(enabled: boolean, intervalMs = 3000): number {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    const timer = window.setInterval(() => setTick((value) => value + 1), intervalMs);
    return () => window.clearInterval(timer);
  }, [enabled, intervalMs]);

  return tick;
}

function liveWebSocketUrl(runId: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/runs/${encodeURIComponent(runId)}/live`;
}

export function useRunWebSocket(
  runId: string | undefined,
  enabled: boolean
): { status: WebSocketStatus; lastEvent: LiveEvent | null; eventCount: number } {
  const [status, setStatus] = useState<WebSocketStatus>(enabled && runId ? 'connecting' : 'disabled');
  const [lastEvent, setLastEvent] = useState<LiveEvent | null>(null);
  const [eventCount, setEventCount] = useState(0);

  useEffect(() => {
    if (!enabled || !runId) {
      setStatus('disabled');
      setLastEvent(null);
      setEventCount(0);
      return undefined;
    }
    let closedByHook = false;
    setStatus('connecting');
    setLastEvent(null);
    setEventCount(0);
    const socket = new WebSocket(liveWebSocketUrl(runId));

    socket.onopen = () => setStatus('connected');
    socket.onerror = () => setStatus('error');
    socket.onclose = () => {
      if (!closedByHook) {
        setStatus('disconnected');
      }
    };
    socket.onmessage = (event: MessageEvent<string>) => {
      try {
        const parsed = JSON.parse(event.data) as LiveEvent;
        setLastEvent(parsed);
        setEventCount((value) => value + 1);
      } catch {
        setStatus('error');
      }
    };

    return () => {
      closedByHook = true;
      socket.close();
    };
  }, [enabled, runId]);

  return { status, lastEvent, eventCount };
}
