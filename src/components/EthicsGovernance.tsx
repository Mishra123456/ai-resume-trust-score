import { motion } from "framer-motion";
import {
  Eye,
  Users,
  Scale,
  Lock,
  FileCheck,
  MessageSquare,
} from "lucide-react";

const principles = [
  {
    icon: Eye,
    title: "Transparency",
    description:
      "All trust calculations are explainable. No black-box scoring—every metric can be traced back to source data.",
  },
  {
    icon: Users,
    title: "Human-in-the-Loop",
    description:
      "Humans always retain final authority. The system supports judgment, not replacement.",
  },
  {
    icon: Scale,
    title: "Fairness",
    description:
      "Trust analytics surface bias signals and uneven performance across decision contexts.",
  },
  {
    icon: Lock,
    title: "Data Privacy",
    description:
      "Uploaded data is processed transiently. No long-term storage or third-party sharing.",
  },
  {
    icon: FileCheck,
    title: "Auditability",
    description:
      "Every insight can be traced to source signals for governance and compliance.",
  },
  {
    icon: MessageSquare,
    title: "Operator Feedback",
    description:
      "Human confidence notes are treated as first-class signals, not noise.",
  },
];

const EthicsGovernance = () => {
  return (
    <section className="py-24 relative overflow-hidden">
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="font-display text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">Ethics & Governance</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            TrustScope is built to support responsible, transparent, and
            human-centered AI adoption.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {principles.map((p, i) => (
            <motion.div
              key={p.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="glass rounded-xl p-6 border border-border/30"
            >
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-lg bg-primary/10">
                  <p.icon className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold mb-1">{p.title}</h3>
                  <p className="text-sm text-muted-foreground">
                    {p.description}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        <p className="text-xs text-muted-foreground text-center mt-12 max-w-2xl mx-auto">
          TrustScope provides decision support, not automated decision-making.
          All insights are advisory and designed for human oversight.
        </p>
      </div>
    </section>
  );
};

export default EthicsGovernance;
