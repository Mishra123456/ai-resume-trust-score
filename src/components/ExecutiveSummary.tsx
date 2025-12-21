import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";

interface ExecutiveSummaryProps {
  text: string; // backend-generated summary
}

export const ExecutiveSummary = ({ text }: ExecutiveSummaryProps) => {
  // Split into sentences for better visual rhythm
  const insights = text
    .split(". ")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => (s.endsWith(".") ? s : `${s}.`));

  return (
    <section className="py-24 relative overflow-hidden">
      {/* Animated Gradient Background */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(135deg, hsl(262 83% 58% / 0.1), hsl(217 91% 60% / 0.1), hsl(187 92% 50% / 0.1))",
          backgroundSize: "400% 400%",
          animation: "gradient-shift 15s ease infinite",
        }}
      />

      {/* Glow Effects */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full bg-purple/10 blur-3xl" />

      <div className="container mx-auto px-6 relative">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-6">
            <Sparkles className="w-4 h-4 text-cyan" />
            <span className="text-sm font-medium">
              AI-Generated Executive Analysis
            </span>
          </div>
          <h2 className="font-display text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">Executive Summary</span>
          </h2>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="max-w-4xl mx-auto"
        >
          <div className="relative p-8 md:p-12 rounded-3xl glass-strong gradient-border overflow-hidden">
            {/* Inner Glow */}
            <div className="absolute inset-0 bg-gradient-radial from-purple/5 via-transparent to-transparent" />

            {/* Rotating Border Glow */}
            <div className="absolute inset-0 rounded-3xl overflow-hidden">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                className="absolute -inset-[100%] bg-gradient-conic from-purple via-blue via-cyan via-blue to-purple opacity-20"
              />
            </div>

            <div className="relative z-10 space-y-6">
              {insights.map((insight, i) => (
                <motion.p
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.3 + i * 0.1, duration: 0.5 }}
                  className="text-lg leading-relaxed text-foreground/90"
                >
                  {insight}
                </motion.p>
              ))}
            </div>

            {/* Decorative Glows */}
            <div className="absolute top-4 right-4 w-20 h-20 rounded-full bg-gradient-to-br from-purple/20 to-cyan/20 blur-2xl" />
            <div className="absolute bottom-4 left-4 w-32 h-32 rounded-full bg-gradient-to-br from-cyan/20 to-blue/20 blur-2xl" />
          </div>
        </motion.div>
      </div>
    </section>
  );
};
