import { motion } from "framer-motion";
import {
  MessageSquare,
  Brain,
  Layers,
  AlertCircle,
  TrendingDown,
  UserX,
  Zap,
} from "lucide-react";

interface AIInsightsPanelProps {
  mlWeights: {
    sentiment_weight: number;
    skepticism_weight: number;
  };
  topRisks: {
    confidence_note: string;
    override_risk: number;
  }[];
  rag: {
    theme: string;
    count: number;
    example: string;
  }[];
}

/* ---------------------------------------
   Reusable UI Components
---------------------------------------- */

const InsightCard = ({
  icon: Icon,
  title,
  color,
  children,
  delay,
}: {
  icon: React.ElementType;
  title: string;
  color: string;
  children: React.ReactNode;
  delay: number;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 40 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ delay, duration: 0.6 }}
    whileHover={{ y: -5 }}
    className="glass rounded-2xl p-6 gradient-border h-full"
  >
    <div
      className={`inline-flex items-center justify-center w-12 h-12 rounded-xl mb-4 ${color}`}
    >
      <Icon className="w-6 h-6" />
    </div>
    <h3 className="text-xl font-bold mb-4">{title}</h3>
    {children}
  </motion.div>
);

const RiskMeter = ({ value }: { value: number }) => (
  <div className="relative w-full h-4 bg-muted rounded-full overflow-hidden">
    <motion.div
      initial={{ width: 0 }}
      whileInView={{ width: `${value}%` }}
      viewport={{ once: true }}
      transition={{ duration: 1, delay: 0.3 }}
      className="h-full rounded-full"
      style={{
        background:
          value > 60
            ? "linear-gradient(90deg, hsl(0, 84%, 60%), hsl(30, 84%, 60%))"
            : value > 30
            ? "linear-gradient(90deg, hsl(30, 84%, 60%), hsl(60, 84%, 60%))"
            : "linear-gradient(90deg, hsl(187, 92%, 50%), hsl(120, 60%, 50%))",
      }}
    />
    <div className="absolute inset-0 flex items-center justify-center text-xs font-bold">
      {value}%
    </div>
  </div>
);

/* ---------------------------------------
   MAIN COMPONENT
---------------------------------------- */

export const AIInsightsPanel = ({
  mlWeights,
  topRisks,
  rag,
}: AIInsightsPanelProps) => {
  const riskScore = Math.round(
    Math.max(
      topRisks.reduce((acc, r) => acc + r.override_risk, 0) /
        Math.max(topRisks.length, 1),
      0
    ) * 100
  );

  return (
    <section className="py-24 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-radial from-cyan/5 via-transparent to-transparent" />

      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="font-display text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">AI-Powered Insights</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            ML predictions and retrieval-based explanations reveal how and why
            trust breaks down in human–AI decision systems
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* NLP / RAG Panel */}
          <InsightCard
            icon={MessageSquare}
            title="Language & RAG Insights"
            color="bg-purple/20 text-purple"
            delay={0.1}
          >
            {rag.map((r) => (
              <div
                key={r.theme}
                className="p-3 rounded-lg bg-muted/50 border border-border/50 mb-3"
              >
                <div className="flex items-center gap-2 mb-1">
                  <AlertCircle className="w-4 h-4 text-cyan" />
                  <span className="font-medium text-sm">{r.theme}</span>
                  <span className="ml-auto text-xs bg-muted px-2 py-0.5 rounded-full">
                    {r.count}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Example: “{r.example}”
                </p>
              </div>
            ))}
          </InsightCard>

          {/* ML Panel */}
          <InsightCard
            icon={Brain}
            title="Predictive ML Insights"
            color="bg-blue/20 text-blue"
            delay={0.2}
          >
            <div className="space-y-6">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">
                    Trust Failure Risk
                  </span>
                  <span
                    className={`text-sm font-bold ${
                      riskScore > 60
                        ? "text-destructive"
                        : riskScore > 30
                        ? "text-yellow-500"
                        : "text-cyan"
                    }`}
                  >
                    {riskScore > 60
                      ? "High"
                      : riskScore > 30
                      ? "Medium"
                      : "Low"}
                  </span>
                </div>
                <RiskMeter value={riskScore} />
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                  <TrendingDown className="w-5 h-5 text-destructive" />
                  <div className="text-sm">
                    <span className="font-medium">Skepticism Influence:</span>
                    <span className="text-muted-foreground ml-1">
                      {mlWeights.skepticism_weight.toFixed(2)}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                  <Zap className="w-5 h-5 text-cyan" />
                  <div className="text-sm">
                    <span className="font-medium">Sentiment Influence:</span>
                    <span className="text-muted-foreground ml-1">
                      {mlWeights.sentiment_weight.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </InsightCard>

          {/* High-Risk Cases */}
          <InsightCard
            icon={Layers}
            title="High-Risk Decisions"
            color="bg-cyan/20 text-cyan"
            delay={0.3}
          >
            {topRisks.map((r, i) => (
              <div
                key={i}
                className="p-3 rounded-lg bg-muted/50 border border-border/50 mb-3"
              >
                <p className="text-sm text-muted-foreground">
                  “{r.confidence_note}”
                </p>
                <div className="mt-2 text-xs font-medium text-cyan">
                  Risk Score: {(r.override_risk * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </InsightCard>
        </div>
      </div>
    </section>
  );
};
