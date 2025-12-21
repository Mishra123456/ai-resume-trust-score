import { motion } from "framer-motion";
import { Shield, Brain, Layers, Activity } from "lucide-react";

const capabilities = [
  {
    icon: Brain,
    title: "Predictive Trust Modeling",
    description:
      "Machine learning models identify patterns that predict when humans are likely to override AI decisions.",
  },
  {
    icon: Layers,
    title: "RAG-Based Explanations",
    description:
      "Retrieval-augmented generation clusters human feedback into interpretable trust themes.",
  },
  {
    icon: Activity,
    title: "Behavioral Analytics",
    description:
      "Tracks override rates, sentiment shifts, and confidence decay over time.",
  },
  {
    icon: Shield,
    title: "Responsible AI Monitoring",
    description:
      "Surfaces early warning signals for trust erosion, bias, and governance risks.",
  },
];

const Capabilities = () => {
  return (
    <section className="py-24 relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-radial from-purple/5 via-transparent to-transparent" />

      <div className="container mx-auto px-6 relative">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="font-display text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">Core Capabilities</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            A unified system combining machine learning, NLP, and analytics to
            understand human trust in AI systems.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {capabilities.map((capability, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, duration: 0.5 }}
              whileHover={{ y: -6 }}
              className="glass rounded-2xl p-6 gradient-border"
            >
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-purple/20 to-cyan/20 mb-4">
                <capability.icon className="w-6 h-6 text-primary" />
              </div>

              <h3 className="text-lg font-semibold mb-2">{capability.title}</h3>

              <p className="text-sm text-muted-foreground leading-relaxed">
                {capability.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Capabilities;
