import { motion } from "framer-motion";
import { Github, Twitter, Linkedin } from "lucide-react";

/* ===============================
   SOCIAL LINKS (EDIT HERE)
================================ */
const SOCIAL_LINKS = [
  {
    icon: Github,
    href: "https://github.com/Mishra123456",
    label: "GitHub",
  },
  {
    icon: Linkedin,
    href: "https://www.linkedin.com/in/mukulmishracs/",
    label: "LinkedIn",
  },
  {
    icon: Twitter,
    href: "https://twitter.com/MukulMishr23858",
    label: "Twitter",
  },
];

const Footer = () => {
  return (
    <footer className="relative py-16 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-t from-navy via-background to-background" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />

      <div className="container mx-auto px-6 relative">
        <div className="flex flex-col items-center text-center">
          {/* Brand */}
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

          {/* Social Links */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1, duration: 0.6 }}
            className="flex gap-4 mb-8"
          >
            {SOCIAL_LINKS.map(({ icon: Icon, href, label }) => (
              <motion.a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={label}
                whileHover={{ scale: 1.1, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="p-3 rounded-full glass border border-border/50 hover:border-primary/50 transition-colors"
              >
                <Icon className="w-5 h-5 text-muted-foreground hover:text-foreground transition-colors" />
              </motion.a>
            ))}
          </motion.div>

          {/* Copyright */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2, duration: 0.6 }}
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

      {/* Bottom Glow */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-96 h-32 bg-gradient-radial from-primary/10 via-transparent to-transparent blur-2xl" />
    </footer>
  );
};

export default Footer;
