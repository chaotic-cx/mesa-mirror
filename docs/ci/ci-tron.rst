CI-tron
=======

`CI-tron <https://ci-tron.dev>`__ is a CI infrastructure for running
a job-specified Linux kernel and firmware set, as well as job-specified user-space
(in a docker image), on a bare-metal test device.

Controlling access to the internet
----------------------------------

By default, test devices do not have access to the internet. This protects
against a whole class of attacks, from botnets to cryptocurrency mining.

In some jobs, it is desirable to allow some limited access to the internet.
Examples include downloading large files that cannot be baked into the
container image used for the job (for instance because there's too many of
them and only a subset is actually useful in each job).

There are two ways of doing this, but in either cases, both the farm admin and
the job must declare that the ``host:port`` target is to be allowed.

For the farm admin, this means editing ``/config/mars_db.yaml``,
and in ``jobs: proxy: allowed_endpoints:`` add ``example.com:443``.

For the job, this means adding a job variable in the form of:

  .. code-block:: yaml

    CI_TRON_PROXY__ALLOWED_ENDPOINTS_CATEGORY__SOME_CATEGORY_NAME__SOME_ENDPOINT_NAME: example.com:443

Cached HTTP requests
````````````````````

When all you need is to download a file from an ``https://...`` URL, and
especially if you want this download to be cached (within what the server
instructs us to do), this is the option you want.

In the job, this requires two things:

- The job's variables need to configure the job's http server to fetch the
  resource and serve it at a specific endpoint:

  .. code-block:: yaml

    CI_TRON_HTTP_ARTIFACT__SOME_NAME__PATH: /some-package.tar.gz
    CI_TRON_HTTP_ARTIFACT__SOME_NAME__URL: https://example.com/some-package-v0.1.tar.gz

- The job's script now needs to fetch ``$CI_TRON_JOB_HTTP_SERVER/some-package.tar.gz``
  to access the resource.

.. note::

  You also can use a regex to avoid having to list a bunch of individual
  files:

  .. code-block:: yaml

    CI_TRON_HTTP_ARTIFACT__SOME_NAME__PATH: /package/(.*)
    CI_TRON_HTTP_ARTIFACT__SOME_NAME__URL: https://example.com/package/\1

  If you want to be able to simply prefix the URL in your client, instead of
  having to change it entirely, you can include the whole URL in the ``PATH``:

  .. code-block:: yaml

    CI_TRON_HTTP_ARTIFACT__SOME_NAME__PATH: /https://s3.freedesktop.org/mesa-tracie-public/(.*)
    CI_TRON_HTTP_ARTIFACT__TRACIE_PUBLIC__URL: https://s3.freedesktop.org/mesa-tracie-public/\1

Pass-through TCP proxy
``````````````````````

For this, you can use the `CONNECT
<https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Methods/CONNECT>`__
proxy available at ``$CI_TRON_JOB_HTTP_SERVER``. The client needs to also
support using a ``CONNECT`` proxy. This involves connecting to
``$CI_TRON_JOB_HTTP_SERVER`` instead of your target, and sending it ``CONNECT
example.com:443 HTTP/1.1\r\n\r\n`` first, then followed by everything the
client would normally do. See this commit for an example of adding that
support: `report-flakes.py: teach it how to use a CONNECT proxy
<https://gitlab.freedesktop.org/mesa/mesa/-/commit/117c57241b3dae38fcee9613596f6227cc85f853>`__.

Reporting test flakes to an IRC channel
---------------------------------------

To enable this, the jobs need to set ``FLAKES_CHANNEL: "#some-channel-name"``
just like for any job, but the farm admin also needs to allow this on their
side, which means:

- Refer to the `Pass-through TCP proxy`_ section above to allow ``irc.oftc.net:6667``.
  Note that ``CI_TRON_PROXY__ALLOWED_ENDPOINTS_CATEGORY__IRC__OFTC: irc.oftc.net:6667``
  is already set in the ``.ci-tron-test`` job (which is inherited by all CI-tron jobs),
  so you don't need to add it again.

- The firewall needs to allow packets going out on that port.
  For that, create ``/config/nftables/outgoing_irc.nft`` with the following contents:

  .. code-block::

     table inet filter {
       chain output {
         tcp dport 6667 accept
       }
     }

  Once done, reload the ``nftables`` config with ``systemctl reload nftables``.
