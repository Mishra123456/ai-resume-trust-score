import { motion } from "framer-motion";
import { Github, Twitter, Linkedin, ExternalLink } from "lucide-react";

const Footer = () => {
  return (
    <footer className="relative py-16 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-t from-navy via-background to-background" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />

      <div className="container mx-auto px-6 relative">
        <div className="flex flex-col items-center text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-8"
          >
            <h3 className="font-display text-3xl font-bold gradient-text mb-2">
              TrustScope
            </h3>
            <p className="text-muted-foreground">
              Applied AI Trust Monitoring Platform
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1, duration: 0.6 }}
            className="flex flex-wrap justify-center gap-8 mb-8"
          >
            {["Documentation", "API Reference", "Research Paper", "Demo"].map(
              (label) => (
                <a
                  key={label}
                  href="#"
                  className="text-muted-foreground hover:text-foreground transition-colors inline-flex items-center gap-1"
                >
                  {label}
                  <ExternalLink className="w-3 h-3" />
                </a>
              )
            )}
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2, duration: 0.6 }}
            className="flex gap-4 mb-8"
          >
            {[Github, Twitter, Linkedin].map((Icon, i) => (
              <motion.a
                key={i}
                href="#"
                whileHover={{ scale: 1.1, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="p-3 rounded-full glass border border-border/50 hover:border-primary/50"
              >
                <Icon className="w-5 h-5 text-muted-foreground hover:text-foreground" />
              </motion.a>
            ))}
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="text-sm text-muted-foreground"
          >
            <p>
              © {new Date().getFullYear()} TrustScope. Built for responsible AI
              adoption.
            </p>
            <p className="mt-1 text-xs">
              Visualizing trust. Predicting failure. Explaining decisions.
            </p>
          </motion.div>
        </div>
      </div>

      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-96 h-32 bg-gradient-radial from-primary/10 via-transparent to-transparent blur-2xl" />
    </footer>
  );
};

export default Footer;
