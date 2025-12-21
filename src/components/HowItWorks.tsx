import { motion } from "framer-motion";
import { Upload, Cpu, BarChart3, FileText, ArrowRight } from "lucide-react";

const steps = [
  {
    icon: Upload,
    number: "01",
    title: "Upload Decision Data",
    description:
      "Drop your CSV with human-AI decision records. We process everything client-side for privacy.",
  },
  {
    icon: Cpu,
    number: "02",
    title: "AI Analysis",
    description:
      "NLP extracts sentiment from notes. ML models predict trust failure risk in real time.",
  },
  {
    icon: BarChart3,
    number: "03",
    title: "Visualize Insights",
    description:
      "Interactive dashboards reveal trust trends, override patterns, and risk zones.",
  },
  {
    icon: FileText,
    number: "04",
    title: "Act on Findings",
    description:
      "Executive summaries and actionable recommendations guide trust improvements.",
  },
];

const HowItWorks = () => {
  return (
    <section className="py-24 relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-radial from-purple/5 via-transparent to-transparent" />

      <div className="container mx-auto px-6 relative">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="font-display text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">How It Works</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            From raw decision data to actionable trust insights in minutes
          </p>
        </motion.div>

        {/* Steps */}
        <div className="relative">
          {/* Desktop connection line */}
          <div className="hidden lg:block absolute top-1/2 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent -translate-y-1/2" />

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {steps.map((step, i) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
                whileHover={{ y: -6 }}
                className="relative"
              >
                <div className="glass rounded-2xl p-6 h-full gradient-border hover:bg-card/80 transition-colors">
                  {/* Number Badge */}
                  <div className="absolute -top-3 -right-3 w-10 h-10 rounded-full bg-gradient-to-br from-purple to-cyan flex items-center justify-center text-sm font-bold shadow-glow-sm">
                    {step.number}
                  </div>

                  {/* Icon */}
                  <div className="inline-flex items-center justify-center w-14 h-14 rounded-xl bg-gradient-to-br from-purple/20 to-cyan/20 mb-4">
                    <step.icon className="w-7 h-7 text-primary" />
                  </div>

                  <h3 className="text-lg font-bold mb-2">{step.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {step.description}
                  </p>
                </div>

                {/* Arrow (desktop only) */}
                {i < steps.length - 1 && (
                  <div className="hidden lg:flex absolute top-1/2 -right-4 -translate-y-1/2 z-10">
                    <ArrowRight className="w-6 h-6 text-primary/50" />
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
