import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ComposedChart,
  Legend,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Shield,
  Users,
} from "lucide-react";

interface AnalyticsDashboardProps {
  metrics: {
    week: string;
    override_rate: number;
    trust_score: number;
    total_cases: number;
  }[];
}

/* ---------------------------------------
   UI Helpers
---------------------------------------- */

const AnimatedCounter = ({
  value,
  suffix = "",
}: {
  value: number;
  suffix?: string;
}) => (
  <span>
    {value.toFixed(suffix === "%" ? 1 : 0)}
    {suffix}
  </span>
);

const StatCard = ({
  title,
  value,
  suffix,
  icon: Icon,
  trend,
  delay,
}: {
  title: string;
  value: number;
  suffix?: string;
  icon: React.ElementType;
  trend?: "up" | "down" | "neutral";
  delay: number;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ delay, duration: 0.5 }}
    whileHover={{ scale: 1.02, y: -5 }}
    className="glass rounded-xl p-6 gradient-border"
  >
    <div className="flex items-start justify-between mb-4">
      <div className="p-3 rounded-lg bg-gradient-to-br from-purple/20 to-cyan/20">
        <Icon className="w-6 h-6 text-primary" />
      </div>
      {trend && (
        <div
          className={`flex items-center gap-1 text-sm ${
            trend === "up"
              ? "text-cyan"
              : trend === "down"
              ? "text-destructive"
              : "text-muted-foreground"
          }`}
        >
          {trend === "up" ? (
            <TrendingUp className="w-4 h-4" />
          ) : trend === "down" ? (
            <TrendingDown className="w-4 h-4" />
          ) : null}
        </div>
      )}
    </div>
    <div className="text-3xl font-bold gradient-text mb-1">
      <AnimatedCounter value={value} suffix={suffix} />
    </div>
    <div className="text-sm text-muted-foreground">{title}</div>
  </motion.div>
);

const ChartCard = ({
  title,
  children,
  delay,
}: {
  title: string;
  children: React.ReactNode;
  delay: number;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ delay, duration: 0.6 }}
    className="glass rounded-2xl p-6 gradient-border"
  >
    <h3 className="text-lg font-semibold mb-6">{title}</h3>
    {children}
  </motion.div>
);

/* ---------------------------------------
   MAIN COMPONENT
---------------------------------------- */

export const AnalyticsDashboard = ({ metrics }: AnalyticsDashboardProps) => {
  const avgTrust =
    metrics.reduce((acc, m) => acc + m.trust_score, 0) /
    Math.max(metrics.length, 1);

  const avgOverride =
    metrics.reduce((acc, m) => acc + m.override_rate, 0) /
    Math.max(metrics.length, 1);

  const totalCases = metrics.reduce((acc, m) => acc + m.total_cases, 0);

  const chartData = metrics.map((m) => ({
    week: m.week,
    trustScore: m.trust_score * 100,
    overrideRate: m.override_rate * 100,
  }));

  const customTooltipStyle = {
    backgroundColor: "hsl(222 47% 11%)",
    border: "1px solid hsl(217 33% 17%)",
    borderRadius: "8px",
    padding: "12px",
  };

  return (
    <section className="py-24 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-radial from-blue/5 via-transparent to-transparent" />

      <div className="container mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="font-display text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">Trust Analytics Dashboard</span>
          </h2>
          <p className="text-muted-foreground text-lg">
            System-level trust trends derived from human–AI interactions
          </p>
        </motion.div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <StatCard
            title="Average Trust Score"
            value={avgTrust * 100}
            suffix="%"
            icon={Shield}
            trend={avgTrust > 0.6 ? "up" : "down"}
            delay={0.1}
          />
          <StatCard
            title="Override Rate"
            value={avgOverride * 100}
            suffix="%"
            icon={Users}
            trend={avgOverride < 0.3 ? "up" : "down"}
            delay={0.2}
          />
          <StatCard
            title="Total Decisions"
            value={totalCases}
            icon={AlertTriangle}
            trend="neutral"
            delay={0.3}
          />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ChartCard title="Trust Score Over Time" delay={0.4}>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient
                    id="trustGradient"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="5%"
                      stopColor="hsl(262, 83%, 58%)"
                      stopOpacity={0.4}
                    />
                    <stop
                      offset="95%"
                      stopColor="hsl(262, 83%, 58%)"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(217, 33%, 17%)"
                />
                <XAxis
                  dataKey="week"
                  stroke="hsl(215, 20%, 65%)"
                  fontSize={12}
                />
                <YAxis
                  stroke="hsl(215, 20%, 65%)"
                  fontSize={12}
                  domain={[0, 100]}
                />
                <Tooltip contentStyle={customTooltipStyle} />
                <Area
                  type="monotone"
                  dataKey="trustScore"
                  stroke="hsl(262, 83%, 58%)"
                  fill="url(#trustGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Override Rate vs Trust" delay={0.5}>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={chartData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(217, 33%, 17%)"
                />
                <XAxis
                  dataKey="week"
                  stroke="hsl(215, 20%, 65%)"
                  fontSize={12}
                />
                <YAxis
                  yAxisId="left"
                  stroke="hsl(215, 20%, 65%)"
                  fontSize={12}
                  domain={[0, 100]}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="hsl(215, 20%, 65%)"
                  fontSize={12}
                  domain={[0, 100]}
                />
                <Tooltip contentStyle={customTooltipStyle} />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="trustScore"
                  stroke="hsl(187, 92%, 50%)"
                  strokeWidth={2}
                  name="Trust Score"
                />
                <Bar
                  yAxisId="right"
                  dataKey="overrideRate"
                  fill="hsl(262, 83%, 58%)"
                  opacity={0.6}
                  name="Override Rate"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      </div>
    </section>
  );
};
