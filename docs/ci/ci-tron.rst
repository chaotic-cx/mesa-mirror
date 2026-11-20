CI-tron
=======

`CI-tron <https://ci-tron.dev>`__ is a CI infrastructure for running
a job-specified Linux kernel and firmware set, as well as job-specified user-space
(in a docker image), on a bare-metal test device.

Reporting test flakes to an IRC channel
---------------------------------------

To enable this, the jobs need to set ``FLAKES_CHANNEL: "#some-channel-name"``
just like for any job, but the farm admin also needs to allow this on their
side, which means:

- The proxy needs to accept connecting to the IRC server we currently use.
  For that, edit ``/config/mars_db.yaml`` and in
  ``jobs: proxy: allowed_endpoints:``, add ``irc.oftc.net:6667``.

- The firewall needs to allow packets going out on that port.
  For that, create ``/config/nftables/outgoing_irc.nft`` with the following contents:

  .. code-block::

     table inet filter {
       chain output {
         tcp dport 6667 accept
       }
     }

  Once done, reload the ``nftables`` config with ``systemctl reload nftables``.
