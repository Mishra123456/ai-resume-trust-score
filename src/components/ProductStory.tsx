import { motion } from "framer-motion";
import {
  AlertTriangle,
  TrendingDown,
  Eye,
  BarChart3,
  Shield,
  Users,
} from "lucide-react";

const reasons = [
  {
    icon: AlertTriangle,
    title: "AI Fails When Humans Stop Trusting",
    description:
      "Even the most accurate AI system becomes useless if operators consistently override its recommendations. Trust is the bridge between capability and adoption.",
    color: "from-purple/20 to-purple/5",
    iconColor: "text-purple",
  },
  {
    icon: TrendingDown,
    title: "Silent Overrides Kill Adoption",
    description:
      "When operators quietly bypass AI recommendations without logging reasons, organizations lose visibility into systemic trust issues until it's too late.",
    color: "from-blue/20 to-blue/5",
    iconColor: "text-blue",
  },
  {
    icon: Eye,
    title: "Trust Must Be Monitored Like Performance",
    description:
      "You measure model accuracy, latency, and uptime. Trust should be treated as an equally critical metric with dashboards, alerts, and trend analysis.",
    color: "from-cyan/20 to-cyan/5",
    iconColor: "text-cyan",
  },
];

const features = [
  { icon: BarChart3, label: "Real-time Analytics" },
  { icon: Shield, label: "Trust Scoring" },
  { icon: Users, label: "Human–AI Alignment" },
];

const ProductStory = () => {
  return (
    <section className="py-24 relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, hsl(var(--muted-foreground)) 1px, transparent 0)`,
            backgroundSize: "40px 40px",
          }}
        />
      </div>

      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="font-display text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">Why Trust Analytics Matters</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            The missing layer in modern AI systems
          </p>
        </motion.div>

        {/* Reason Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          {reasons.map((reason, i) => (
            <motion.div
              key={reason.title}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, duration: 0.6 }}
              whileHover={{ y: -10, scale: 1.02 }}
              className="relative group"
            >
              <div
                className={`absolute inset-0 rounded-2xl bg-gradient-to-b ${reason.color} opacity-0 group-hover:opacity-100 transition-opacity duration-500`}
              />
              <div className="relative glass rounded-2xl p-8 h-full gradient-border">
                <motion.div
                  whileHover={{ rotate: 10, scale: 1.1 }}
                  className={`inline-flex items-center justify-center w-14 h-14 rounded-xl bg-gradient-to-br ${reason.color} mb-6`}
                >
                  <reason.icon className={`w-7 h-7 ${reason.iconColor}`} />
                </motion.div>
                <h3 className="text-xl font-bold mb-3">{reason.title}</h3>
                <p className="text-muted-foreground leading-relaxed">
                  {reason.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Feature Pills */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.4, duration: 0.6 }}
          className="flex flex-wrap justify-center gap-4"
        >
          {features.map((feature, i) => (
            <motion.div
              key={feature.label}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.5 + i * 0.1, duration: 0.4 }}
              whileHover={{ scale: 1.05 }}
              className="inline-flex items-center gap-3 px-6 py-3 rounded-full glass gradient-border"
            >
              <feature.icon className="w-5 h-5 text-cyan" />
              <span className="font-medium">{feature.label}</span>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default ProductStory;
