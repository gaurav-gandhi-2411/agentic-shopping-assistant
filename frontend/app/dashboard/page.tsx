"use client"

import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api/client"
import type { BrandStat, DashboardData, OccasionStat, PairingStat } from "@/lib/api/types"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MetricCard({
  label,
  value,
  sub,
}: {
  label: string
  value: string
  sub?: string
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-3xl">{value}</CardTitle>
      </CardHeader>
      {sub && (
        <CardContent>
          <p className="text-xs text-muted-foreground">{sub}</p>
        </CardContent>
      )}
    </Card>
  )
}

function PairingsTable({ rows }: { rows: PairingStat[] }) {
  if (rows.length === 0) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Top pairings</CardTitle>
        <CardDescription>
          Category pairs ordered by &ldquo;Add the Look&rdquo; count (minimum 5 signals)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="pb-2 pr-4 font-medium">Anchor</th>
                <th className="pb-2 pr-4 font-medium">Fill</th>
                <th className="pb-2 pr-4 font-medium">Occasion</th>
                <th className="pb-2 pr-4 font-medium text-right">Add the Look</th>
                <th className="pb-2 font-medium text-right">Total signals</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-b last:border-0">
                  <td className="py-2 pr-4">{r.anchor_category}</td>
                  <td className="py-2 pr-4">{r.fill_category}</td>
                  <td className="py-2 pr-4 capitalize">{r.occasion}</td>
                  <td className="py-2 pr-4 text-right font-medium">{r.add_the_look}</td>
                  <td className="py-2 text-right text-muted-foreground">{r.total_signals}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}

function SegmentTable({
  title,
  description,
  labelKey,
  rows,
}: {
  title: string
  description: string
  labelKey: "occasion" | "brand"
  rows: (OccasionStat | BrandStat)[]
}) {
  if (rows.length === 0) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="pb-2 pr-4 font-medium capitalize">{labelKey}</th>
                <th className="pb-2 pr-4 font-medium text-right">Looks shown</th>
                <th className="pb-2 pr-4 font-medium text-right">Add-the-Look rate</th>
                <th className="pb-2 font-medium text-right">Basket delta (INR)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const label = labelKey === "occasion"
                  ? (r as OccasionStat).occasion
                  : (r as BrandStat).brand
                const delta = r.basket_delta_inr
                return (
                  <tr key={i} className="border-b last:border-0">
                    <td className="py-2 pr-4 capitalize">{label}</td>
                    <td className="py-2 pr-4 text-right">{r.looks_shown}</td>
                    <td className="py-2 pr-4 text-right">
                      {(r.add_the_look_rate * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 text-right">
                      {delta !== null
                        ? `${delta >= 0 ? "+" : ""}${delta.toFixed(0)}`
                        : "—"}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <p className="text-lg font-medium text-muted-foreground">
        No data yet &mdash; start styling to see metrics
      </p>
      <p className="mt-2 text-sm text-muted-foreground">
        Events appear here after users interact with outfit suggestions.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  const { data, isLoading, isError } = useQuery<DashboardData>({
    queryKey: ["dashboard"],
    queryFn: () => api.dashboard.get(),
    // Dashboard data is aggregate — 60s stale time is fine.
    staleTime: 60_000,
    retry: 2,
  })

  const hasData = data && data.looks_shown > 0

  return (
    <main className="min-h-screen bg-background p-6 md:p-10">
      <div className="mx-auto max-w-5xl space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Brand Dashboard</h1>
          <p className="mt-1 text-muted-foreground">
            Aggregate styling metrics from flywheel event data.
            Updates in real time as users interact with outfit suggestions.
          </p>
        </div>

        {/* Loading / error states */}
        {isLoading && (
          <p className="text-sm text-muted-foreground">Loading metrics&hellip;</p>
        )}
        {isError && (
          <p className="text-sm text-destructive">
            Could not load dashboard metrics. The backend may be unavailable.
          </p>
        )}

        {/* No-data empty state */}
        {data && !hasData && <EmptyState />}

        {/* Metrics — only render when there is data */}
        {hasData && (
          <>
            {/* Key metric cards */}
            <section className="grid gap-4 sm:grid-cols-3">
              <MetricCard
                label="Looks shown"
                value={data.looks_shown.toLocaleString()}
              />
              <MetricCard
                label="Add-the-Look rate"
                value={`${(data.add_the_look_rate * 100).toFixed(1)}%`}
                sub="Share of looks where the user added the full outfit"
              />
              <MetricCard
                label="Add-single rate"
                value={`${(data.add_single_rate * 100).toFixed(1)}%`}
                sub="Baseline — single-item adds for context"
              />
            </section>

            {/* Basket size section */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  Basket size: look vs single-item
                </CardTitle>
                <CardDescription>
                  Mean look_total_inr per event type
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid gap-4 sm:grid-cols-3">
                  <div>
                    <p className="text-xs text-muted-foreground">Look avg (INR)</p>
                    <p className="text-2xl font-semibold">
                      {data.basket_size.look_avg_inr !== null
                        ? `₹${data.basket_size.look_avg_inr.toLocaleString("en-IN", {
                            maximumFractionDigits: 0,
                          })}`
                        : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Single-item avg (INR)</p>
                    <p className="text-2xl font-semibold">
                      {data.basket_size.single_avg_inr !== null
                        ? `₹${data.basket_size.single_avg_inr.toLocaleString("en-IN", {
                            maximumFractionDigits: 0,
                          })}`
                        : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Delta (look &minus; single)</p>
                    <p className="text-2xl font-semibold">
                      {data.basket_size.delta_inr !== null
                        ? `${data.basket_size.delta_inr >= 0 ? "+" : ""}₹${Math.abs(
                            data.basket_size.delta_inr
                          ).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`
                        : "—"}
                    </p>
                  </div>
                </div>
                <p className="text-xs italic text-muted-foreground">
                  {data.basket_size.caveat}
                </p>
              </CardContent>
            </Card>

            {/* Top pairings */}
            {data.top_pairings.length > 0 && (
              <PairingsTable rows={data.top_pairings} />
            )}

            {/* By occasion */}
            {data.by_occasion.length > 0 && (
              <SegmentTable
                title="By occasion"
                description="Add-the-Look rate and basket delta segmented by occasion tag"
                labelKey="occasion"
                rows={data.by_occasion}
              />
            )}

            {/* By brand */}
            {data.by_brand.length > 0 && (
              <SegmentTable
                title="By brand"
                description="Add-the-Look rate and basket delta segmented by brand"
                labelKey="brand"
                rows={data.by_brand}
              />
            )}
          </>
        )}
      </div>
    </main>
  )
}
